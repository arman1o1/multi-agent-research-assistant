"""ADK Runner wrapper — manages sessions, runs the pipeline, and yields events."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.genai import types

from src.agents.pipeline import create_pipeline
from src.config import get_settings

logger = logging.getLogger(__name__)

APP_NAME = "research_assistant"


@dataclass
class ResearchEvent:
    """A structured event from the research pipeline."""

    event_type: str  # "agent_event" | "pipeline_status"
    agent: str = ""
    stage: str = ""
    status: str = ""  # "running" | "complete" | "error"
    content: str = ""
    timestamp: str = ""
    data: dict[str, Any] = field(default_factory=dict)


class ResearchRunner:
    """Wraps ADK Runner for the research pipeline."""

    def __init__(self) -> None:
        settings = get_settings()

        # Ensure data directory exists for SQLite
        os.makedirs("data", exist_ok=True)

        # Use DatabaseSessionService with SQLite for persistence
        try:
            self._session_service = DatabaseSessionService(db_url=settings.db_url)
        except Exception as e:
            logger.warning(f"Failed to init DatabaseSessionService: {e}. Using in-memory.")
            self._session_service = InMemorySessionService()

        self._pipeline = create_pipeline()
        self._runner = Runner(
            agent=self._pipeline,
            app_name=APP_NAME,
            session_service=self._session_service,
        )
        self._approvals: dict[str, asyncio.Event] = {}
        self._approval_decisions: dict[str, str] = {}

    def create_approval_event(self, session_id: str) -> asyncio.Event:
        """Create a new approval event for a session."""
        event = asyncio.Event()
        self._approvals[session_id] = event
        # Do not pop from self._approval_decisions here to avoid clearing pre-resolved approvals
        return event

    def resolve_approval(self, session_id: str, decision: str) -> None:
        """Resolve a pending approval event."""
        self._approval_decisions[session_id] = decision
        if session_id in self._approvals:
            self._approvals[session_id].set()

    async def wait_for_approval(self, session_id: str) -> str:
        """Wait for an approval decision."""
        if session_id in self._approval_decisions:
            self._approvals.pop(session_id, None)
            return self._approval_decisions.pop(session_id, "rejected")

        event = self._approvals.get(session_id)
        if not event:
            return "rejected"
        await event.wait()
        # Clean up
        self._approvals.pop(session_id, None)
        return self._approval_decisions.pop(session_id, "rejected")

    async def run_research(
        self,
        topic: str,
        user_id: str | None = None,
        session_id: str | None = None,
        mode: str = "auto",
    ) -> AsyncGenerator[ResearchEvent, None]:
        """Run the research pipeline on a topic, yielding events.

        Args:
            topic: The research topic to investigate.
            user_id: Optional user ID for session management.
            session_id: Optional session ID. Created if not provided.
            mode: Approval mode ("auto" or "approval").

        Yields:
            ResearchEvent objects representing pipeline progress.
        """
        user_id = user_id or f"user_{uuid.uuid4().hex[:8]}"
        session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"

        # Create or get session
        session = await self._session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )

        if session is None:
            session = await self._session_service.create_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=session_id,
                state={
                    # Seed empty defaults for template variables referenced in agent
                    # instructions that won't exist on the first loop iteration.
                    "critique": "",
                    "findings_list": "[]",
                },
            )

        # Build the user message
        user_message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=f"Research this topic: {topic}")],
        )

        yield ResearchEvent(
            event_type="pipeline_status",
            status="started",
            content=f"Starting research on: {topic}",
            data={"session_id": session_id, "user_id": user_id, "topic": topic},
        )

        current_agent = ""
        completed = False

        try:
            async for event in self._runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=user_message,
            ):
                if completed:
                    # Let the generator drain without emitting more events
                    continue

                # Track which agent is producing events
                author = getattr(event, "author", "") or ""

                if author and author != current_agent:
                    # Agent transition
                    if current_agent:
                        yield ResearchEvent(
                            event_type="agent_event",
                            agent=current_agent,
                            status="complete",
                            content=f"{current_agent} finished.",
                        )

                        # Pause for approval after planner finishes in approval mode
                        if current_agent == "planner" and mode == "approval":
                            session_state = await self.get_session_state(user_id, session_id)
                            plan_content = session_state.get("research_plan", "No plan generated.")
                            yield ResearchEvent(
                                event_type="pipeline_status",
                                stage="Research Plan",
                                status="awaiting_approval",
                                content=plan_content,
                                data={"plan": plan_content, "session_id": session_id},
                            )
                            self.create_approval_event(session_id)
                            decision = await self.wait_for_approval(session_id)
                            if decision == "rejected":
                                yield ResearchEvent(
                                    event_type="pipeline_status",
                                    status="error",
                                    content="Research plan was rejected by the user.",
                                )
                                completed = True
                                break

                    current_agent = author
                    logger.info(f"Agent transition: {current_agent}")
                    yield ResearchEvent(
                        event_type="agent_event",
                        agent=current_agent,
                        status="running",
                        content=f"{current_agent} started working...",
                    )

                # Extract text content from the event
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        text = getattr(part, "text", None)
                        if text:
                            yield ResearchEvent(
                                event_type="agent_event",
                                agent=author,
                                status="running",
                                content=text,
                            )

                        # Surface tool calls in the activity feed
                        fn_call = getattr(part, "function_call", None)
                        if fn_call:
                            tool_name = getattr(fn_call, "name", "tool")
                            yield ResearchEvent(
                                event_type="agent_event",
                                agent=author,
                                status="running",
                                content=f"Calling {tool_name}...",
                            )

                # Only treat the writer's final response as pipeline completion.
                # Each sub-agent in a SequentialAgent emits is_final_response(),
                # so we must filter to the last agent in the sequence.
                if (
                    hasattr(event, "is_final_response")
                    and event.is_final_response()
                    and author == "writer"
                ):
                    final_text = ""
                    if event.content and event.content.parts:
                        final_text = "\n".join(
                            p.text for p in event.content.parts if getattr(p, "text", None)
                        )

                    yield ResearchEvent(
                        event_type="pipeline_status",
                        status="complete",
                        content="Research complete.",
                        data={"report": final_text, "session_id": session_id},
                    )
                    completed = True

            # If we exhausted the event stream without seeing the writer,
            # pull the report from session state (set via output_key="final_report").
            if not completed:
                logger.info("Event stream ended — checking session state for report")
                session = await self._session_service.get_session(
                    app_name=APP_NAME, user_id=user_id, session_id=session_id,
                )
                final_report = ""
                if session and session.state:
                    final_report = session.state.get("final_report", "")

                if current_agent:
                    yield ResearchEvent(
                        event_type="agent_event",
                        agent=current_agent,
                        status="complete",
                        content=f"{current_agent} finished.",
                    )

                yield ResearchEvent(
                    event_type="pipeline_status",
                    status="complete",
                    content="Research complete.",
                    data={"report": final_report, "session_id": session_id},
                )

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            yield ResearchEvent(
                event_type="pipeline_status",
                status="error",
                content=f"Pipeline error: {e}",
            )

    async def get_session_state(
        self, user_id: str, session_id: str
    ) -> dict[str, Any]:
        """Get the current state of a research session."""
        session = await self._session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )

        if session is None:
            return {"error": "Session not found"}

        return dict(session.state)


async def run_research(topic: str) -> str:
    """Convenience function: run research and return the final report.

    Used by the evaluation framework.
    """
    runner = ResearchRunner()
    final_report = ""

    async for event in runner.run_research(topic):
        if event.event_type == "pipeline_status" and event.status == "complete":
            final_report = event.data.get("report", "")

    return final_report
