"""Tests for tool modules: search, documents, and state tools."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# --- Search Tool Tests ---


class TestSearchWeb:
    """Tests for the search_web function."""

    @patch("src.tools.search._get_provider", return_value="duckduckgo")
    @patch("src.tools.search._search_duckduckgo")
    def test_duckduckgo_search_returns_results(self, mock_ddg, mock_provider):
        """DuckDuckGo search returns normalized results."""
        mock_ddg.return_value = [
            {
                "title": "Test Article",
                "url": "https://example.com",
                "snippet": "A test snippet",
                "source": "duckduckgo",
            }
        ]

        from src.tools.search import search_web

        result = json.loads(search_web("test query", max_results=1))

        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Test Article"
        assert result["results"][0]["source"] == "duckduckgo"
        assert result["provider"] == "duckduckgo"

    @patch("src.tools.search._get_provider", return_value="tavily")
    @patch("src.tools.search._search_tavily")
    def test_tavily_search_returns_results(self, mock_tavily, mock_provider):
        """Tavily search returns normalized results."""
        mock_tavily.return_value = [
            {
                "title": "Tavily Result",
                "url": "https://tavily.com",
                "snippet": "Tavily content",
                "source": "tavily",
            }
        ]

        from src.tools.search import search_web

        result = json.loads(search_web("test query"))

        assert result["provider"] == "tavily"
        assert len(result["results"]) == 1

    @patch("src.tools.search._get_provider", return_value="tavily")
    @patch("src.tools.search._search_tavily", side_effect=Exception("API error"))
    @patch("src.tools.search._search_duckduckgo")
    def test_tavily_failure_falls_back_to_duckduckgo(
        self, mock_ddg, mock_tavily, mock_provider
    ):
        """When Tavily fails, falls back to DuckDuckGo."""
        mock_ddg.return_value = [
            {"title": "Fallback", "url": "https://ddg.co", "snippet": "FB", "source": "duckduckgo"}
        ]

        from src.tools.search import search_web

        result = json.loads(search_web("test"))

        assert len(result["results"]) == 1
        mock_ddg.assert_called_once()

    @patch("src.tools.search._get_provider", return_value="duckduckgo")
    @patch("src.tools.search._search_duckduckgo", side_effect=Exception("Network error"))
    def test_search_failure_returns_error(self, mock_ddg, mock_provider):
        """Search failure returns error in JSON."""
        from src.tools.search import search_web

        result = json.loads(search_web("test"))

        assert "error" in result


# --- Document Tool Tests ---


class TestReadWebpage:
    """Tests for the read_webpage function."""

    @patch("trafilatura.fetch_url", return_value=None)
    def test_fetch_failure_returns_error(self, mock_fetch):
        """Returns error message when URL can't be fetched."""
        from src.tools.documents import read_webpage

        result = read_webpage("https://nonexistent.example.com")
        assert "Error" in result

    @patch("trafilatura.extract", return_value="Extracted content here")
    @patch("trafilatura.fetch_url", return_value="<html>content</html>")
    def test_successful_extraction(self, mock_fetch, mock_extract):
        """Returns extracted text on success."""
        from src.tools.documents import read_webpage

        result = read_webpage("https://example.com")
        assert result == "Extracted content here"

    @patch("trafilatura.extract", return_value="x" * 20000)
    @patch("trafilatura.fetch_url", return_value="<html>long content</html>")
    def test_content_truncation(self, mock_fetch, mock_extract):
        """Long content is truncated to max_content_chars."""
        from src.tools.documents import read_webpage

        result = read_webpage("https://example.com")
        assert "truncated" in result


class TestReadPdf:
    """Tests for the read_pdf function."""

    @patch("pymupdf.open", side_effect=FileNotFoundError("not found"))
    def test_missing_file_returns_error(self, mock_open):
        """Returns error for missing PDF file."""
        from src.tools.documents import read_pdf

        result = read_pdf("./nonexistent/file.pdf")
        assert "Error" in result

    def test_path_traversal_returns_error(self):
        """Returns access denied for files outside the workspace."""
        from src.tools.documents import read_pdf

        result = read_pdf("../../../sensitive_file.pdf")
        assert "Access denied" in result


# --- State Tool Tests ---


class TestStateTools:
    """Tests for state management tools."""

    def _make_tool_context(self) -> MagicMock:
        """Create a mock ToolContext with a dict-backed state."""
        ctx = MagicMock()
        ctx.state = {}
        ctx.actions = MagicMock()
        return ctx

    def test_save_finding_creates_list(self):
        """First save_finding creates the findings list."""
        from src.tools.state_tools import save_finding

        ctx = self._make_tool_context()
        result = save_finding("Title", "Content", "https://src.com", ctx)

        assert "1" in result
        findings = json.loads(ctx.state["findings_list"])
        assert len(findings) == 1
        assert findings[0]["title"] == "Title"

    def test_save_finding_appends(self):
        """Subsequent save_finding calls append to existing list."""
        from src.tools.state_tools import save_finding

        ctx = self._make_tool_context()
        save_finding("First", "Content1", "https://a.com", ctx)
        save_finding("Second", "Content2", "https://b.com", ctx)

        findings = json.loads(ctx.state["findings_list"])
        assert len(findings) == 2

    def test_get_findings_empty(self):
        """get_findings returns message when no findings exist."""
        from src.tools.state_tools import get_findings

        ctx = self._make_tool_context()
        result = get_findings(ctx)
        assert "No findings" in result

    def test_get_findings_returns_data(self):
        """get_findings returns saved findings as JSON."""
        from src.tools.state_tools import get_findings, save_finding

        ctx = self._make_tool_context()
        save_finding("Test", "Data", "https://x.com", ctx)
        result = get_findings(ctx)

        parsed = json.loads(result)
        assert len(parsed) == 1

    def test_escalate_research_sets_flag(self):
        """escalate_research sets actions.escalate to True."""
        from src.tools.state_tools import escalate_research

        ctx = self._make_tool_context()
        result = escalate_research("Research is sufficient", ctx)

        assert ctx.actions.escalate is True
        assert ctx.state["escalation_reason"] == "Research is sufficient"
        assert "ended" in result.lower()
