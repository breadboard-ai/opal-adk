"""Sets up the agent execution environment and manages the agent sessions."""

from collections.abc import AsyncGenerator, Mapping
import os
from absl import logging
from google.adk import runners
from google.adk.agents import loop_agent
from google.adk.agents import sequential_agent
from google.adk.events import event
from google.adk.memory import in_memory_memory_service
from google.adk.sessions import in_memory_session_service
from google.genai import types
from opal_adk import flags
from opal_adk.agents import node_agent
from opal_adk.agents import report_writing_agent
from opal_adk.agents import research_agent
from opal_adk.data_model import agent_step
from opal_adk.data_model import opal_plan_step
from opal_adk.types import ui_type

_PARAMETER_KEY = "query"
_MAX_ITERATIONS = 10


def _create_content_from_string(content: str) -> types.Content:
  return types.Content(role="user", parts=[types.Part(text=content)])


def _extract_input_parameter(plan_step: opal_plan_step.OpalPlanStep) -> str:
  """Extracts a single input parameter key from an OpalPlanStep.

  Args:
    plan_step: The OpalPlanStep containing the input parameters.

  Returns:
    The extracted input parameter key.

  Raises:
    ValueError: If there isn't exactly one input parameter.
  """
  input_params = plan_step.input_parameters
  if len(input_params) != 1:
    raise ValueError(
        "Executor.py: When parameter_name is not provided, expected exactly "
        f"one input parameter, but got {len(input_params)}: {input_params}"
    )
  return input_params[0]


class AgentExecutor:
  """Executes various agent workflows using in-memory session and memory services.

  This class provides methods to run different agent-based tasks, such as
  deep research and report generation, by orchestrating agents and managing
  their execution context.
  """

  def __init__(
      self,
      *,
      project_id: str | None = None,
      location: str | None = None,
      genai_api_key: str | None = None,
  ):
    """Initializes the AgentExecutor with optional project and location configurations.

    The AgentExecutor can be initialized with optional project and location
    configurations. If both are provided then they will be used to configure
    the environment variables GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.
    If neither are provided then the environment variables GOOGLE_CLOUD_PROJECT,
    GOOGLE_CLOUD_LOCATION, and GOOGLE_GENAI_USE_VERTEXAI must be set. If only
    one is provided then an exception will be raised.

    Args:
      project_id: The Google Cloud project ID to use for Vertex AI. If provided,
        `location` must also be provided. If not provided then there must be an
        existing environment variable set for GOOGLE_CLOUD_PROJECT.
      location: The Google Cloud location (e.g., "us-central1") to use for
        Vertex AI. If provided, `project_id` must also be provided. If not
        provided then there must be an existing environment variable set for
        GOOGLE_CLOUD_LOCATION.
      genai_api_key: API key for using Gemini API instead of Vertex AI.

    Raises:
      ValueError: If only one of `project_id` or `location` is provided.
    """

    if bool(project_id) != bool(location):
      raise ValueError(
          "Both project_id and location must be provided, but got"
          f" project_id={project_id!r} and location={location!r}"
      )

    if genai_api_key:
      os.environ["GOOGLE_API_KEY"] = genai_api_key
      os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"
    elif project_id and location:
      os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
      os.environ["GOOGLE_CLOUD_LOCATION"] = location
      os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
    else:
      # Neither project_id nor location were provided,
      # so check for required env vars.
      required_env_vars = [
          "GOOGLE_CLOUD_PROJECT",
          "GOOGLE_CLOUD_LOCATION",
          "GOOGLE_GENAI_USE_VERTEXAI",
      ]
      missing_vars = [var for var in required_env_vars if var not in os.environ]
      if missing_vars:
        raise ValueError(
            "When project_id and location are not provided, the following "
            f"environment variables must be set: {', '.join(missing_vars)}"
        )
    if flags.get_debug_logging():
      logging.set_verbosity(logging.DEBUG)

    self.session_service = in_memory_session_service.InMemorySessionService()
    self.memory_service = in_memory_memory_service.InMemoryMemoryService()
    logging.info("AgentExecutor: %r created.", self)

  def __repr__(self) -> str:
    return (
        f"{self.__class__.__name__}(session_service={self.session_service!r}, "
        f"memory_service={self.memory_service!r})"
    )

  async def execute_agent_node(
      self,
      *,
      user_id: str,
      step: agent_step.AgentStep,
      session_id: str | None = None,
      execution_inputs: Mapping[str, types.Content] | None = None,
  ) -> AsyncGenerator[event.Event, None] | None:
    """Executes a Breadboard node in agent mode.

    This execution method will create and execute an agent representing a single
    node in a Breadboard graph that has "agent" mode enabled. These nodes
    are similar to non-agent modes but instructions and context are provided
    by the front end and they have access to more tools making them more
    flexible but less deterministic.

    Args:
      user_id: The ID of the user.
      step: The opal plan step configuration. The `step_name` is used for the
        agent's name.
      session_id: The session id to use when starting a new session or resuming
        a previous session. If a session id is not provided chat features will
        not be available.
      execution_inputs: The inputs provided for the agent execution. This can
        contain additional input elements such as input images or file paths.

    Returns:
      Chunks of the agent's output as the execution progresses.
    """
    logging.info(
        "AgentExecutor: Node agent execution called with input: %s", step
    )
    logging.info(
        "AgentExecutor: Node agent execution called with execution_inputs: %s",
        execution_inputs,
    )
    logging.info("executor: model_constraint: %s", step.model_constraint)
    agent = node_agent.node_agent(step.model_constraint, ui_type=step.ui_type)
    orchestrator_agent = loop_agent.LoopAgent(
        name="opal_adk_node_agent_orchestrator",
        description=(
            "Loop agent that executes the node agent until the objective is"
            " completed or the agent cannot continue and fails."
        ),
        sub_agents=[agent],
        max_iterations=_MAX_ITERATIONS,
    )
    if not session_id and step.ui_type == ui_type.UIType.CHAT:
      raise ValueError(
          "Executor: session_id must be provided for chat UI type."
      )

    # Create a new session if a session with this id doesn't yet exist.
    if not await self.session_service.get_session(
        app_name=step.step_name, user_id=user_id, session_id=session_id
    ):
      await self.session_service.create_session(
          app_name=step.step_name, user_id=user_id, session_id=session_id
      )

    runner = runners.Runner(
        app_name=step.step_name,
        agent=orchestrator_agent,
        session_service=self.session_service,
        memory_service=self.memory_service,
    )

    return runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=step.objective,
        invocation_id=step.invocation_id,
    )

  async def execute_deep_research_agent(
      self,
      user_id: str,
      opal_step: opal_plan_step.OpalPlanStep,
      *,
      execution_inputs: Mapping[str, types.Content],
      session_id: str | None = None,
  ) -> AsyncGenerator[event.Event, None] | None:
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
      execution_inputs: The inputs provided for the agent execution. This will
        include inputs such as the research topic.
      session_id: The session id to use when starting a new session or resuming
        a previous session.

    Returns:
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
    if session_id:
      # Create a new session if a session with this id doesn't yet exist.
      if not await self.session_service.get_session(
          app_name=opal_step.step_name,
          user_id=user_id,
          session_id=session_id,
      ):
        session = await self.session_service.create_session(
            app_name=opal_step.step_name,
            user_id=user_id,
            session_id=session_id,
        )
      else:
        session = await self.session_service.get_session(
            app_name=opal_step.step_name,
            user_id=user_id,
            session_id=session_id,
        )
    else:
      session = await self.session_service.create_session(
          app_name=opal_step.step_name, user_id=user_id
      )
    runner = runners.Runner(
        app_name=opal_step.step_name,
        agent=deep_research_agent,
        session_service=self.session_service,
        memory_service=self.memory_service,
    )

    input_param = _extract_input_parameter(opal_step)
    if input_param not in execution_inputs:
      raise ValueError(
          f"Executor.py: Input parameter '{input_param}' not found in execution"
          f" inputs. Available inputs: {list(execution_inputs.keys())}"
      )

    research_query = execution_inputs[input_param]
    assert session is not None
    return runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=research_query,
    )
