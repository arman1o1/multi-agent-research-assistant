"""LLM-as-judge evaluator using Gemini.

Sends a research report + topic to Gemini and receives structured scores
across four quality dimensions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from google import genai
from google.genai import types

from src.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DimensionScore:
    """Score and justification for a single evaluation dimension."""
    score: int          # 1-5
    justification: str


@dataclass
class EvaluationResult:
    """Full evaluation output for one research report."""
    topic: str
    factual_accuracy: DimensionScore
    comprehensiveness: DimensionScore
    coherence: DimensionScore
    citation_quality: DimensionScore
    overall_score: float = field(init=False)

    def __post_init__(self) -> None:
        self.overall_score = round(
            (
                self.factual_accuracy.score
                + self.comprehensiveness.score
                + self.coherence.score
                + self.citation_quality.score
            )
            / 4,
            2,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Rubric (system prompt for the judge LLM)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are an expert research report evaluator. You will receive a TOPIC and a
RESEARCH REPORT (in Markdown). Your job is to evaluate the report on exactly
four dimensions, assigning an integer score from 1 to 5 for each, along with
a concise justification (1-3 sentences).

## Scoring Rubric

### 1. Factual Accuracy
How well are the claims in the report supported by the cited sources?
- **5 — Excellent**: All major claims are accurate and directly supported by
  cited, verifiable sources. No factual errors detected.
- **4 — Good**: Nearly all claims are accurate; minor unsupported statements
  exist but do not mislead the reader.
- **3 — Adequate**: Most claims are accurate, but several statements lack
  source support or contain minor inaccuracies.
- **2 — Poor**: Multiple factual errors or unsupported claims that could
  mislead the reader.
- **1 — Very Poor**: Pervasive inaccuracies; the report cannot be trusted
  as a factual resource.

### 2. Comprehensiveness
Does the report cover the topic thoroughly, addressing key subtopics,
perspectives, and nuances?
- **5 — Excellent**: Covers all major facets of the topic, including
  historical context, current state, differing viewpoints, and future
  outlook where relevant.
- **4 — Good**: Covers most important aspects; only minor subtopics are
  missing.
- **3 — Adequate**: Covers the basics but misses important subtopics or
  perspectives.
- **2 — Poor**: Significant gaps in coverage; feels superficial.
- **1 — Very Poor**: Barely scratches the surface; would not inform a
  reader new to the topic.

### 3. Coherence
Is the report well-structured, logically organized, and easy to follow?
- **5 — Excellent**: Clear introduction, logically sequenced sections,
  smooth transitions, and a strong conclusion. The narrative flows
  naturally.
- **4 — Good**: Well-organized overall; minor structural issues or abrupt
  transitions.
- **3 — Adequate**: Understandable but somewhat disjointed; the reader has
  to work to follow the argument.
- **2 — Poor**: Disorganized; sections feel randomly ordered or
  repetitive.
- **1 — Very Poor**: Incoherent; no discernible structure or logical
  flow.

### 4. Citation Quality
Are sources diverse, reliable, and properly attributed?
- **5 — Excellent**: Uses a diverse set of high-quality sources (academic
  papers, reputable news outlets, official reports). All sources are
  properly cited with titles/URLs.
- **4 — Good**: Good source diversity and reliability; minor attribution
  issues.
- **3 — Adequate**: Sources are present but lack diversity (e.g., all from
  one type) or some citations are incomplete.
- **2 — Poor**: Few sources; questionable reliability or missing
  attribution for key claims.
- **1 — Very Poor**: No citations, or sources are unreliable / fabricated.

## Output Format
Respond with a JSON object matching this schema exactly:
{
  "factual_accuracy":  {"score": <int 1-5>, "justification": "<string>"},
  "comprehensiveness": {"score": <int 1-5>, "justification": "<string>"},
  "coherence":         {"score": <int 1-5>, "justification": "<string>"},
  "citation_quality":  {"score": <int 1-5>, "justification": "<string>"}
}
Do NOT include any text outside the JSON object.
"""

# JSON schema for structured output enforcement
_EVAL_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "factual_accuracy": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "score": types.Schema(type=types.Type.INTEGER),
                "justification": types.Schema(type=types.Type.STRING),
            },
            required=["score", "justification"],
        ),
        "comprehensiveness": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "score": types.Schema(type=types.Type.INTEGER),
                "justification": types.Schema(type=types.Type.STRING),
            },
            required=["score", "justification"],
        ),
        "coherence": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "score": types.Schema(type=types.Type.INTEGER),
                "justification": types.Schema(type=types.Type.STRING),
            },
            required=["score", "justification"],
        ),
        "citation_quality": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "score": types.Schema(type=types.Type.INTEGER),
                "justification": types.Schema(type=types.Type.STRING),
            },
            required=["score", "justification"],
        ),
    },
    required=["factual_accuracy", "comprehensiveness", "coherence", "citation_quality"],
)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ReportEvaluator:
    """Evaluates research reports using Gemini as an LLM judge."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._client = genai.Client(api_key=self._settings.google_api_key)
        self._model = self._settings.evaluator_model

    async def evaluate(self, topic: str, report: str) -> EvaluationResult:
        """Score a research report on the four quality dimensions.

        Args:
            topic: The original research topic/question.
            report: The full research report as a Markdown string.

        Returns:
            An ``EvaluationResult`` with per-dimension scores and an overall
            average.
        """
        user_message = (
            f"## Topic\n{topic}\n\n"
            f"## Research Report\n{report}"
        )

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=_EVAL_RESPONSE_SCHEMA,
                temperature=0.0,
            ),
        )

        raw: dict[str, Any] = json.loads(response.text)
        logger.debug("Raw judge response for '%s': %s", topic, raw)

        return EvaluationResult(
            topic=topic,
            factual_accuracy=DimensionScore(**raw["factual_accuracy"]),
            comprehensiveness=DimensionScore(**raw["comprehensiveness"]),
            coherence=DimensionScore(**raw["coherence"]),
            citation_quality=DimensionScore(**raw["citation_quality"]),
        )
