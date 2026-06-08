"""State management tools and LoopAgent escalation tool."""

from __future__ import annotations

import json
import logging
from typing import Any

from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)


def save_finding(
    title: str,
    content: str,
    source_url: str,
    tool_context: ToolContext,
) -> str:
    """Save a research finding to shared state.

    Args:
        title: Brief title of the finding.
        content: The finding content/summary.
        source_url: URL of the source where this was found.
        tool_context: ADK tool context (injected automatically).

    Returns:
        Confirmation message with the number of findings saved so far.
    """
    findings: list[dict[str, Any]] = json.loads(
        tool_context.state.get("findings_list", "[]")
    )

    findings.append(
        {
            "title": title,
            "content": content,
            "source_url": source_url,
        }
    )

    tool_context.state["findings_list"] = json.dumps(findings)
    return f"Finding saved. Total findings: {len(findings)}"


def get_findings(tool_context: ToolContext) -> str:
    """Retrieve all saved research findings from shared state.

    Args:
        tool_context: ADK tool context (injected automatically).

    Returns:
        JSON string of all findings, or a message if none exist.
    """
    findings_json = tool_context.state.get("findings_list", "[]")
    findings = json.loads(findings_json)

    if not findings:
        return "No findings saved yet."

    return json.dumps(findings, indent=2)


def escalate_research(reason: str, tool_context: ToolContext) -> str:
    """Signal that the research is sufficient and the loop should end.

    Call this when the research findings adequately cover the topic and no
    further research iterations are needed.

    Args:
        reason: Explanation of why the research is considered sufficient.
        tool_context: ADK tool context (injected automatically).

    Returns:
        Confirmation that escalation was triggered.
    """
    tool_context.state["escalation_reason"] = reason
    tool_context.actions.escalate = True
    logger.info(f"Research loop escalated: {reason}")
    return f"Research loop ended. Reason: {reason}"
