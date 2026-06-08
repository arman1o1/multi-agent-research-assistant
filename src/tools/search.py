"""Web search tool with Tavily primary + DuckDuckGo fallback."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.config import get_settings

logger = logging.getLogger(__name__)

_provider_logged = False


def _get_provider() -> str:
    """Determine and log the active search provider (once)."""
    global _provider_logged
    provider = get_settings().search_provider
    if not _provider_logged:
        logger.info(f"Search provider: {provider}")
        _provider_logged = True
    return provider


def _search_tavily(query: str, max_results: int) -> list[dict[str, Any]]:
    """Search using Tavily API."""
    from tavily import TavilyClient

    settings = get_settings()
    client = TavilyClient(api_key=settings.tavily_api_key)
    response = client.search(query=query, max_results=max_results)

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
            "source": "tavily",
        }
        for r in response.get("results", [])
    ]


def _search_duckduckgo(query: str, max_results: int) -> list[dict[str, Any]]:
    """Search using DuckDuckGo (free, no API key)."""
    from ddgs import DDGS

    results = []
    ddgs = DDGS()
    for r in ddgs.text(query, max_results=max_results):
        results.append(
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
                "source": "duckduckgo",
            }
        )
    return results


_tavily_warned = False


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information on a topic.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 5).

    Returns:
        JSON string containing search results with title, url, and snippet.
    """
    global _tavily_warned
    provider = _get_provider()

    try:
        if provider == "tavily":
            results = _search_tavily(query, max_results)
        else:
            results = _search_duckduckgo(query, max_results)
    except Exception as e:
        # If Tavily fails, try DuckDuckGo as emergency fallback
        if provider == "tavily":
            if not _tavily_warned:
                logger.warning(f"Tavily unavailable ({e}), falling back to DuckDuckGo")
                _tavily_warned = True
            try:
                results = _search_duckduckgo(query, max_results)
            except Exception as e2:
                logger.error(f"DuckDuckGo fallback also failed: {e2}")
                return json.dumps(
                    {
                        "error": f"Tavily error ({e}) and DuckDuckGo fallback error ({e2})",
                        "results": [],
                    }
                )
        else:
            logger.error(f"Search failed with {provider}: {e}")
            return json.dumps({"error": str(e), "results": []})

    return json.dumps({"results": results, "provider": provider, "query": query})
