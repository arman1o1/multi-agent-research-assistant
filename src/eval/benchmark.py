"""Batch evaluation runner for the Multi-Agent Research Assistant.

Loads topics, runs the research pipeline on each, evaluates the output,
and produces a summary table (printed + saved as CSV).

Usage:
    python -m src.eval.benchmark                      # run all topics
    python -m src.eval.benchmark --complexity hard     # filter by complexity
    python -m src.eval.benchmark --indices 0 3 7       # run specific indices
    python -m src.eval.benchmark --output results.csv  # custom output path
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

from src.api.runner import run_research
from src.eval.evaluator import EvaluationResult, ReportEvaluator

logger = logging.getLogger(__name__)

TOPICS_FILE = Path(__file__).parent / "topics.json"
DEFAULT_OUTPUT = Path("benchmark_results.csv")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_topics(
    path: Path = TOPICS_FILE,
    *,
    complexity: str | None = None,
    indices: Sequence[int] | None = None,
) -> list[dict]:
    """Load and optionally filter topics from the JSON file.

    Args:
        path: Path to ``topics.json``.
        complexity: If provided, keep only topics matching this level
            (``easy``, ``medium``, or ``hard``).
        indices: If provided, keep only topics at these 0-based positions
            (applied *after* the complexity filter).

    Returns:
        List of topic dicts with ``topic``, ``complexity``, and ``domain``
        keys.
    """
    with open(path, encoding="utf-8") as f:
        topics: list[dict] = json.load(f)

    if complexity:
        topics = [t for t in topics if t["complexity"] == complexity]

    if indices:
        topics = [topics[i] for i in indices if i < len(topics)]

    return topics


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

async def _run_single(
    topic_entry: dict,
    evaluator: ReportEvaluator,
) -> tuple[EvaluationResult, float]:
    """Run the pipeline + evaluation for a single topic.

    Returns:
        ``(EvaluationResult, latency_seconds)``
    """
    topic_str: str = topic_entry["topic"]
    logger.info("▶ Running topic: %s", topic_str)

    t0 = time.perf_counter()
    report = await run_research(topic_str)
    latency = time.perf_counter() - t0

    logger.info("  Pipeline finished in %.1fs, evaluating…", latency)
    result = await evaluator.evaluate(topic_str, report)
    return result, latency


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

async def run_benchmark(
    topics: list[dict],
    *,
    output_path: Path = DEFAULT_OUTPUT,
) -> list[tuple[EvaluationResult, float]]:
    """Run the full benchmark: pipeline → evaluate → report.

    Args:
        topics: List of topic dicts to benchmark.
        output_path: Where to write the CSV results.

    Returns:
        List of ``(EvaluationResult, latency_seconds)`` tuples.
    """
    evaluator = ReportEvaluator()
    results: list[tuple[EvaluationResult, float]] = []

    for entry in topics:
        try:
            result, latency = await _run_single(entry, evaluator)
            results.append((result, latency))
        except Exception:
            logger.exception("✗ Failed on topic: %s", entry["topic"])

    # --- Print summary table ------------------------------------------------
    _print_summary(results)

    # --- Save CSV -----------------------------------------------------------
    _save_csv(results, output_path)
    logger.info("Results saved to %s", output_path)

    return results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

_HEADER = (
    "Topic",
    "Domain",
    "Complexity",
    "Factual",
    "Comprehensive",
    "Coherence",
    "Citations",
    "Overall",
    "Latency (s)",
)


_TOPICS_CACHE: dict[str, dict] = {}


def _get_topic_entry(topic: str) -> dict:
    global _TOPICS_CACHE
    if not _TOPICS_CACHE:
        try:
            with open(TOPICS_FILE, encoding="utf-8") as f:
                for t in json.load(f):
                    _TOPICS_CACHE[t["topic"]] = t
        except Exception:
            pass
    return _TOPICS_CACHE.get(topic, {})


def _fmt_row(
    result: EvaluationResult,
    latency: float,
    topic_entry: dict | None = None,
) -> tuple[str, ...]:
    entry = topic_entry or _get_topic_entry(result.topic)
    return (
        result.topic[:60],
        entry.get("domain", "-"),
        entry.get("complexity", "-"),
        str(result.factual_accuracy.score),
        str(result.comprehensiveness.score),
        str(result.coherence.score),
        str(result.citation_quality.score),
        f"{result.overall_score:.2f}",
        f"{latency:.1f}",
    )


def _print_summary(results: list[tuple[EvaluationResult, float]]) -> None:
    """Pretty-print the results table to stdout."""
    if not results:
        print("No results to display.")
        return

    col_widths = [len(h) for h in _HEADER]
    rows: list[tuple[str, ...]] = []
    for res, lat in results:
        row = _fmt_row(res, lat)
        rows.append(row)
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    sep = "  "
    header_line = sep.join(h.ljust(w) for h, w in zip(_HEADER, col_widths))
    divider = sep.join("-" * w for w in col_widths)

    print()
    print(header_line)
    print(divider)
    for row in rows:
        print(sep.join(c.ljust(w) for c, w in zip(row, col_widths)))

    # Averages
    if len(results) > 1:
        avg_scores = [
            sum(r.factual_accuracy.score for r, _ in results) / len(results),
            sum(r.comprehensiveness.score for r, _ in results) / len(results),
            sum(r.coherence.score for r, _ in results) / len(results),
            sum(r.citation_quality.score for r, _ in results) / len(results),
            sum(r.overall_score for r, _ in results) / len(results),
            sum(lat for _, lat in results) / len(results),
        ]
        print(divider)
        avg_cells = (
            "AVERAGE",
            "",
            "",
            f"{avg_scores[0]:.2f}",
            f"{avg_scores[1]:.2f}",
            f"{avg_scores[2]:.2f}",
            f"{avg_scores[3]:.2f}",
            f"{avg_scores[4]:.2f}",
            f"{avg_scores[5]:.1f}",
        )
        print(sep.join(c.ljust(w) for c, w in zip(avg_cells, col_widths)))
    print()


def _save_csv(
    results: list[tuple[EvaluationResult, float]],
    path: Path,
) -> None:
    """Write results to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_HEADER)
        for res, lat in results:
            writer.writerow(_fmt_row(res, lat))


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Multi-Agent Research Assistant benchmark.",
    )
    parser.add_argument(
        "--complexity",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Filter topics by complexity level.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="0-based indices of topics to run (after complexity filter).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


async def _async_main(args: argparse.Namespace) -> None:
    topics = load_topics(complexity=args.complexity, indices=args.indices)
    if not topics:
        print("No topics matched the given filters.", file=sys.stderr)
        sys.exit(1)

    print(f"Running benchmark on {len(topics)} topic(s)…\n")
    await run_benchmark(topics, output_path=args.output)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
