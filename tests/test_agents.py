"""Tests for agent creation and pipeline assembly."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.agents import (
    critic as critic_mod,
    fact_checker as fact_checker_mod,
    planner as planner_mod,
    researcher as researcher_mod,
    writer as writer_mod,
)


def _mock_settings(**overrides) -> MagicMock:
    """Create a mock Settings with sensible defaults."""
    s = MagicMock()
    s.planner_model = overrides.get("planner_model", "gemini-2.5-pro")
    s.researcher_model = overrides.get("researcher_model", "gemini-2.5-flash")
    s.critic_model = overrides.get("critic_model", "gemini-2.5-flash")
    s.fact_checker_model = overrides.get("fact_checker_model", "gemini-2.5-flash")
    s.writer_model = overrides.get("writer_model", "gemini-2.5-pro")
    s.max_research_iterations = overrides.get("max_research_iterations", 3)
    return s


class TestAgentCreation:
    """Tests that agents are created with correct configuration."""

    def test_planner_agent_config(self):
        """Planner uses pro model and has no tools."""
        with patch.object(planner_mod, "get_settings", return_value=_mock_settings()):
            agent = planner_mod.create_planner_agent()

        assert agent.name == "planner"
        assert agent.model == "gemini-2.5-pro"
        assert agent.output_key == "research_plan"
        assert not agent.tools

    def test_researcher_agent_has_tools(self):
        """Researcher has search, read, and state tools."""
        with patch.object(researcher_mod, "get_settings", return_value=_mock_settings()):
            agent = researcher_mod.create_researcher_agent()

        assert agent.name == "researcher"
        assert agent.model == "gemini-2.5-flash"
        assert agent.output_key == "raw_findings"
        assert len(agent.tools) == 4

    def test_critic_agent_has_escalate_tool(self):
        """Critic has the escalate_research tool."""
        with patch.object(critic_mod, "get_settings", return_value=_mock_settings()):
            agent = critic_mod.create_critic_agent()

        assert agent.name == "critic"
        assert agent.output_key == "critique"
        assert len(agent.tools) == 1

    def test_fact_checker_has_search(self):
        """Fact-checker has search tool for verification."""
        with patch.object(fact_checker_mod, "get_settings", return_value=_mock_settings()):
            agent = fact_checker_mod.create_fact_checker_agent()

        assert agent.name == "fact_checker"
        assert agent.output_key == "fact_check_report"
        assert len(agent.tools) == 1

    def test_writer_agent_no_tools(self):
        """Writer is pure synthesis, no tools."""
        with patch.object(writer_mod, "get_settings", return_value=_mock_settings()):
            agent = writer_mod.create_writer_agent()

        assert agent.name == "writer"
        assert agent.model == "gemini-2.5-pro"
        assert agent.output_key == "final_report"
        assert not agent.tools


class TestPipelineAssembly:
    """Tests for the full pipeline structure."""

    def test_pipeline_structure(self):
        """Pipeline has correct workflow structure."""
        mock_s = _mock_settings()

        with (
            patch.object(planner_mod, "get_settings", return_value=mock_s),
            patch.object(researcher_mod, "get_settings", return_value=mock_s),
            patch.object(critic_mod, "get_settings", return_value=mock_s),
            patch.object(fact_checker_mod, "get_settings", return_value=mock_s),
            patch.object(writer_mod, "get_settings", return_value=mock_s),
        ):
            # Import here to avoid module-level side effects
            from src.agents.pipeline import create_pipeline

            # Patch get_settings in the pipeline module too
            with patch("src.agents.pipeline.get_settings", return_value=mock_s):
                pipeline = create_pipeline()

        from google.adk import Workflow

        assert isinstance(pipeline, Workflow)
        assert pipeline.name == "research_pipeline"

        # Verify key workflow nodes are in the graph edges
        edges_str = str(pipeline.edges)
        assert "planner" in edges_str
        assert "researcher" in edges_str
        assert "critic" in edges_str
        assert "fact_checker" in edges_str
        assert "writer" in edges_str

