"""FastAPI server — REST endpoints, SSE streaming, and WebSocket for approval mode."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.api.export import export_pdf, markdown_to_html
from src.api.runner import ResearchEvent, ResearchRunner
from src.config import get_settings

logger = logging.getLogger(__name__)

# --- App Setup ---

app = FastAPI(
    title="Multi-Agent Research Assistant",
    description="A multi-agent system that produces comprehensive, cited research reports.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- State ---

# Active research sessions: session_id -> {user_id, topic, status, report, events}
sessions: dict[str, dict[str, Any]] = {}

# WebSocket connections for approval mode: session_id -> WebSocket
approval_connections: dict[str, WebSocket] = {}

# Background tasks for research: session_id -> asyncio.Task
session_tasks: dict[str, asyncio.Task] = {}

# Runner instance (lazy init)
_runner: ResearchRunner | None = None


def get_runner() -> ResearchRunner:
    """Get or create the ResearchRunner singleton."""
    global _runner
    if _runner is None:
        _runner = ResearchRunner()
    return _runner


# --- Request/Response Models ---


class ResearchRequest(BaseModel):
    topic: str
    mode: str = "auto"  # "auto" or "approval"


class ResearchResponse(BaseModel):
    session_id: str
    status: str
    topic: str


class SessionStatus(BaseModel):
    session_id: str
    status: str
    topic: str
    report: str | None = None
    event_count: int = 0


# --- Helper ---


def _event_to_sse(event: ResearchEvent) -> str:
    """Convert a ResearchEvent to an SSE-formatted string."""
    data = {
        "type": event.event_type,
        "agent": event.agent,
        "stage": event.stage,
        "status": event.status,
        "content": event.content,
        "timestamp": event.timestamp or datetime.now(timezone.utc).isoformat(),
        "data": event.data,
    }
    return f"data: {json.dumps(data)}\n\n"


def _slugify_topic(topic: str) -> str:
    """Sanitize the topic to create a safe, clean filename slug."""
    cleaned = re.sub(r"[^\w\s-]", "", topic.lower())
    slug = re.sub(r"[-\s]+", "_", cleaned)
    slug = slug.strip("_")
    return slug or "research_report"


# --- Endpoints ---


@app.post("/api/research", response_model=ResearchResponse)
async def start_research(request: ResearchRequest) -> ResearchResponse:
    """Start a new research session."""
    session_id = f"session_{uuid.uuid4().hex[:12]}"
    user_id = f"user_{uuid.uuid4().hex[:8]}"

    sessions[session_id] = {
        "user_id": user_id,
        "topic": request.topic,
        "mode": request.mode,
        "status": "started",
        "report": None,
        "events": [],
    }

    # Launch the pipeline in the background and track the task
    task = asyncio.create_task(_run_pipeline(session_id, user_id, request.topic, request.mode))
    session_tasks[session_id] = task

    return ResearchResponse(
        session_id=session_id,
        status="started",
        topic=request.topic,
    )


async def _run_pipeline(session_id: str, user_id: str, topic: str, mode: str = "auto") -> None:
    """Background task: run the research pipeline and store events."""
    runner = get_runner()

    try:
        async for event in runner.run_research(
            topic=topic,
            user_id=user_id,
            session_id=session_id,
            mode=mode,
        ):
            event.timestamp = datetime.now(timezone.utc).isoformat()

            if session_id in sessions:
                sessions[session_id]["events"].append(event)

                if event.event_type == "pipeline_status":
                    if event.status == "complete":
                        sessions[session_id]["status"] = "complete"
                        sessions[session_id]["report"] = event.data.get("report", "")
                    elif event.status == "error":
                        sessions[session_id]["status"] = "error"

    except asyncio.CancelledError:
        logger.info(f"Pipeline cancelled for {session_id}")
        if session_id in sessions:
            sessions[session_id]["status"] = "error"
            sessions[session_id]["events"].append(
                ResearchEvent(
                    event_type="pipeline_status",
                    status="error",
                    content="Research cancelled by user.",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )
        raise
    except Exception as e:
        logger.error(f"Pipeline failed for {session_id}: {e}", exc_info=True)
        if session_id in sessions:
            sessions[session_id]["status"] = "error"
            sessions[session_id]["events"].append(
                ResearchEvent(
                    event_type="pipeline_status",
                    status="error",
                    content=str(e),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )
    finally:
        session_tasks.pop(session_id, None)


@app.get("/api/research/{session_id}", response_model=SessionStatus)
async def get_research_status(session_id: str) -> SessionStatus:
    """Get the status of a research session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    return SessionStatus(
        session_id=session_id,
        status=session["status"],
        topic=session["topic"],
        report=session.get("report"),
        event_count=len(session["events"]),
    )


@app.get("/api/research/{session_id}/stream")
async def stream_research(session_id: str) -> StreamingResponse:
    """SSE endpoint for real-time research progress."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    async def event_generator():
        """Yield SSE events as they arrive."""
        last_index = 0

        while True:
            session = sessions.get(session_id)
            if session is None:
                break

            events = session["events"]

            # Send any new events since last check
            while last_index < len(events):
                event = events[last_index]
                yield _event_to_sse(event)
                last_index += 1

                # If pipeline is complete or errored, send final event and stop
                if (
                    event.event_type == "pipeline_status"
                    and event.status in ("complete", "error")
                ):
                    return

            # Poll interval — short enough for responsive UI
            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/research/{session_id}/report")
async def get_report(session_id: str) -> Response:
    """Download the research report as Markdown."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    report = session.get("report")
    topic = session.get("topic", "report")

    if not report:
        raise HTTPException(status_code=404, detail="Report not available yet")

    filename = f"report_{_slugify_topic(topic)}.md"
    return Response(
        content=report,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )


@app.get("/api/research/{session_id}/report/pdf")
async def get_report_pdf(session_id: str) -> Response:
    """Download the research report as PDF."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    report = session.get("report")
    topic = session.get("topic", "report")

    if not report:
        raise HTTPException(status_code=404, detail="Report not available yet")

    try:
        pdf_bytes = bytes(export_pdf(report))
        filename = f"report_{_slugify_topic(topic)}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            },
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/research/{session_id}/report/html")
async def get_report_html(session_id: str) -> Response:
    """Download the research report as HTML."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    report = session.get("report")
    topic = session.get("topic", "report")

    if not report:
        raise HTTPException(status_code=404, detail="Report not available yet")

    html_content = markdown_to_html(report)
    filename = f"report_{_slugify_topic(topic)}.html"
    return Response(
        content=html_content,
        media_type="text/html",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )


# --- Approval & Cancellation Endpoints ---


class ApprovalRequest(BaseModel):
    approved: bool


@app.patch("/api/research/{session_id}")
async def patch_approval(session_id: str, request: ApprovalRequest) -> dict[str, str]:
    """Approve or reject a pending research plan via HTTP PATCH."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    decision = "approved" if request.approved else "rejected"
    sessions[session_id]["approval_status"] = decision

    # Resolve the runner's pending approval event
    get_runner().resolve_approval(session_id, decision)

    # Notify websocket client if connected
    ws = approval_connections.get(session_id)
    if ws:
        try:
            await ws.send_json({"type": "approval_ack", "status": decision})
        except Exception:
            pass

    return {"status": decision}


@app.post("/api/research/{session_id}/cancel")
async def cancel_research(session_id: str) -> dict[str, str]:
    """Cancel a running research session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # Cancel the asyncio task
    task = session_tasks.get(session_id)
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Resolve any pending approvals so it doesn't hang
    get_runner().resolve_approval(session_id, "rejected")

    # Update status to error/cancelled
    if session_id in sessions:
        sessions[session_id]["status"] = "error"
        has_final = any(
            e.event_type == "pipeline_status" and e.status in ("complete", "error")
            for e in sessions[session_id]["events"]
        )
        if not has_final:
            sessions[session_id]["events"].append(
                ResearchEvent(
                    event_type="pipeline_status",
                    status="error",
                    content="Research cancelled by user.",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )

    return {"status": "cancelled"}


# --- WebSocket for Approval Mode ---


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """WebSocket for approval mode — bidirectional communication."""
    origin = websocket.headers.get("origin")
    settings = get_settings()
    allowed_origins = {
        f"http://localhost:{settings.port}",
        f"http://127.0.0.1:{settings.port}",
        f"http://{settings.host}:{settings.port}",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    }
    host_header = websocket.headers.get("host")
    if host_header:
        allowed_origins.add(f"http://{host_header}")
        allowed_origins.add(f"https://{host_header}")

    if origin and origin not in allowed_origins:
        logger.warning(f"WebSocket rejected: origin={origin}, host={host_header}")
        await websocket.close(code=1008)
        return

    await websocket.accept()
    approval_connections[session_id] = websocket
    logger.info(f"WebSocket connected for session {session_id}")

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            msg_type = data.get("type")

            # Determine decision
            decision = None
            if action == "approve":
                decision = "approved"
            elif action == "reject":
                decision = "rejected"
            elif msg_type == "approval":
                decision = "approved" if data.get("approved") is True else "rejected"

            if decision:
                logger.info(f"Research plan {decision} via WebSocket for {session_id}")
                if session_id in sessions:
                    sessions[session_id]["approval_status"] = decision
                
                # Resolve the runner's pending approval event
                get_runner().resolve_approval(session_id, decision)
                
                await websocket.send_json(
                    {"type": "approval_ack", "status": decision}
                )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {session_id}")
    finally:
        approval_connections.pop(session_id, None)


# --- Static Files (Frontend) ---

# Mount frontend static files — must be last to not shadow API routes
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


# --- Entry Point ---


def main() -> None:
    """Run the server."""
    import uvicorn

    settings = get_settings()
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    logger.info(f"Search provider: {settings.search_provider}")
    logger.info(f"Approval mode: {settings.approval_mode}")

    uvicorn.run(
        "src.api.server:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        reload_dirs=["src"],
        reload_excludes=[".venv", "__pycache__", "*.pyc"],
    )


if __name__ == "__main__":
    main()
