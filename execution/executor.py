"""Sets up the agent execution environment and manages the agent sessions."""

from collections.abc import AsyncGenerator
import logging
from google.adk import runners
from google.adk.agents import sequential_agent
from google.adk.memory import in_memory_memory_service
from google.adk.sessions import in_memory_session_service
from google.genai import types
from opal_adk.agents import report_writing_agent
from opal_adk.agents import research_agent
from opal_adk.data_model import opal_plan_step


class AgentExecutor:
  """Executes various agent workflows using in-memory session and memory services.

  This class provides methods to run different agent-based tasks, such as
  deep research and report generation, by orchestrating agents and managing
  their execution context.
  """

  def __init__(self):
    self.session_service = in_memory_session_service.InMemorySessionService()
    self.memory_service = in_memory_memory_service.InMemoryMemoryService()
    logging.info("AgentExecutor: %r created.", self)

  def __repr__(self) -> str:
    return (
        f"{self.__class__.__name__}(session_service={self.session_service!r}, "
        f"memory_service={self.memory_service!r})"
    )

  def _create_content_from_string(self, content: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=content)])

  async def execute_deep_research_agent(
      self, user_id: str, opal_step: opal_plan_step.OpalPlanStep
  ) -> AsyncGenerator[str, None] | None:
    """Executes an agent workflow for in-depth research and report generation.

    This method orchestrates a `SequentialAgent` composed of a `research_agent`
    and a `report_writing_agent`. The `research_agent` performs iterative data
    gathering and analysis based on a predefined plan. Subsequently, the
    `report_writing_agent` compiles the findings from the research phase into
    a comprehensive report.

    Args:
      user_id: The ID of the user initiating the research.
      opal_step: The opal plan step configuration, containing agent parameters
        such as iterations.

    Yields:
      Chunks of the agent's output as the execution progresses, typically
      including research findings and report sections.
    """
    deep_research_agent = sequential_agent.SequentialAgent(
        name="deep_research_agent",
        description=(
            "Performs deep research based on a query and generates an"
            " output report on its findings."
        ),
        sub_agents=[
            research_agent.deep_research_agent(iterations=opal_step.iterations),
            report_writing_agent.report_writing_agent(
                parent_agent_output_key=research_agent.OUTPUT_KEY
            ),
        ],
    )
    session = await self.session_service.create_session(
        app_name=opal_step.step_name, user_id=user_id
    )
    runner = runners.Runner(
        app_name=opal_step.step_name,
        agent=deep_research_agent,
        session_service=self.session_service,
        memory_service=self.memory_service,
    )

    return runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=self._create_content_from_string(opal_step.step_intent),
    )
