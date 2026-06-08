"""Tests for the FastAPI server endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.server import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestResearchEndpoints:
    """Tests for the research API endpoints."""

    def test_start_research(self, client):
        """POST /api/research creates a session."""
        response = client.post(
            "/api/research",
            json={"topic": "Quantum computing", "mode": "auto"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["status"] == "started"
        assert data["topic"] == "Quantum computing"

    def test_get_nonexistent_session(self, client):
        """GET /api/research/invalid returns 404."""
        response = client.get("/api/research/nonexistent_session")
        assert response.status_code == 404

    def test_get_report_before_completion(self, client):
        """Report endpoint returns 404 before pipeline completes."""
        # Start a research session
        start_response = client.post(
            "/api/research",
            json={"topic": "Test topic"},
        )
        session_id = start_response.json()["session_id"]

        # Report won't be ready immediately
        response = client.get(f"/api/research/{session_id}/report")
        assert response.status_code == 404

    def test_get_report_pdf_before_completion(self, client):
        """PDF endpoint returns 404 before pipeline completes."""
        start_response = client.post(
            "/api/research",
            json={"topic": "Test topic"},
        )
        session_id = start_response.json()["session_id"]

        response = client.get(f"/api/research/{session_id}/report/pdf")
        assert response.status_code == 404


class TestExport:
    """Tests for the export module."""

    def test_markdown_to_html(self):
        """Markdown converts to styled HTML."""
        from src.api.export import markdown_to_html

        html = markdown_to_html("# Hello\n\nThis is a **test**.")
        assert "<h1" in html
        assert "<strong>test</strong>" in html
        assert "Inter" in html  # font reference
