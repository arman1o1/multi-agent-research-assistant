"""Pipeline assembly — graph-based Workflow orchestration."""

from __future__ import annotations

from google.adk import Context, Workflow
from google.adk.events import Event
from google.adk.workflow import node

from src.agents.critic import create_critic_agent
from src.agents.fact_checker import create_fact_checker_agent
from src.agents.planner import create_planner_agent
from src.agents.researcher import create_researcher_agent
from src.agents.writer import create_writer_agent
from src.config import get_settings


@node
def route_research(ctx: Context):
    """Routes execution based on research adequacy and iteration count."""
    state = ctx.state
    settings = get_settings()
    iter_count = state.get("research_iteration", 0)
    is_escalated = "escalation_reason" in state and bool(state["escalation_reason"])

    if is_escalated or iter_count >= settings.max_research_iterations:
        yield Event(route="complete")
    else:
        state["research_iteration"] = iter_count + 1
        yield Event(route="continue")


def create_pipeline() -> Workflow:
    """Assemble the full research pipeline.

    Pipeline flow:
        Planner → Researcher → Critic → route_research (loop back to Researcher or proceed) → Fact-Checker → Writer
    """
    planner = create_planner_agent()
    researcher = create_researcher_agent()
    critic = create_critic_agent()
    fact_checker = create_fact_checker_agent()
    writer = create_writer_agent()

    # Define the graph-based workflow replacing SequentialAgent and LoopAgent
    pipeline = Workflow(
        name="research_pipeline",
        description="Multi-agent research pipeline that produces comprehensive, cited reports.",
        edges=[
            ("START", planner, researcher, critic, route_research),
            (route_research, {"continue": researcher, "complete": fact_checker}),
            (fact_checker, writer),
        ],
    )

    return pipeline


# Export root_agent for ADK CLI compatibility (`adk run`, `adk web`)
root_agent = create_pipeline()

