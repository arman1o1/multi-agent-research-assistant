"""Multi-Agent Research Assistant — Configuration."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    google_api_key: str = ""
    tavily_api_key: str = ""

    # Model assignments — Pro for planning/writing, Flash for the rest
    planner_model: str = "gemini-2.5-pro"
    researcher_model: str = "gemini-2.5-flash"
    critic_model: str = "gemini-2.5-flash"
    fact_checker_model: str = "gemini-2.5-flash"
    writer_model: str = "gemini-2.5-pro"
    evaluator_model: str = "gemini-2.5-flash"

    # Pipeline parameters
    max_research_iterations: int = 3
    max_search_results: int = 5
    max_content_chars: int = 15000  # truncation limit for document content
    approval_mode: bool = False

    # Database
    db_url: str = "sqlite+aiosqlite:///./data/sessions.db"

    # Server
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def search_provider(self) -> str:
        """Determine search provider based on available API keys."""
        if self.tavily_api_key:
            return "tavily"
        return "duckduckgo"

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()

    # Set the Google API key in the environment for ADK to pick up
    if settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key

    return settings
