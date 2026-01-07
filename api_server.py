"""A FastAPI server for the OPAL ADK."""

from collections.abc import AsyncGenerator, Mapping, Sequence
import logging

from absl import app
from absl import flags
from typing import NewType
import fastapi
from fastapi import responses
from opal_adk import flags as opal_flags
from opal_adk.data_model import opal_plan_step
from opal_adk.execution import executor
import pydantic
import uvicorn

# Define flags for host and port
_HOST = flags.DEFINE_string("host", "localhost", "The host address to bind to.")
_PORT = flags.DEFINE_integer(
    "port", 8000, "The port to listen on.", lower_bound=1024, upper_bound=65535
)
# Initialize FastAPI app
fast_api_app = fastapi.FastAPI()
router = fastapi.APIRouter()
UserId = NewType("UserId", str)
Model = NewType("Model", str)
Query = NewType("Query", str)


class ExecuteAgentRequest(pydantic.BaseModel):
  """Request body for the /execute_agent endpoint."""

  model: Model
  query: Query
  agent_parameters: Mapping[str, str]
  user_id: UserId
  iterations: int


class OpalAdkApi:
  """Provides API methods for interacting with the OPAL ADK agents.

  This class initializes an AgentExecutor and provides methods to execute
  different agents, such as the DeepResearch Agent.
  """

  def __init__(self):
    project_id = opal_flags.get_project_id()
    location = opal_flags.get_location()
    self.executor = executor.AgentExecutor(
        project_id=project_id, location=location
    )

  async def execute_deep_research_agent(
      self, request: ExecuteAgentRequest
  ) -> AsyncGenerator[str, None]:
    """Executes the DeepResearch Agent with the given request.

    Args:
      request: An ExecuteAgentRequest containing the parameters for the agent.

    Yields:
      A string representing events or outputs from the agent execution.
    """
    logging.info("Executing DeepResearch Agent with request: %r", request)
    opal_step = opal_plan_step.OpalPlanStep(
        step_name="DeepResearch Agent",
        step_intent=request.query,
        model_api=request.model,
        iterations=request.iterations,
    )
    agent_generator = await self.executor.execute_deep_research_agent(
        user_id=request.user_id, opal_step=opal_step
    )
    if agent_generator is not None:
      async for event in agent_generator:
        yield f"response: {event.model_dump_json()}\n\n"


@router.post("/execute_deep_research_agent")
async def execute_deep_research_agent(
    request: ExecuteAgentRequest,
    agent_api: OpalAdkApi = fastapi.Depends(OpalAdkApi),
) -> responses.StreamingResponse:
  """API endpoint to execute a command on an agent."""
  logging.info("ApiServer: Received execute_agent request: %s", request)
  return responses.StreamingResponse(
      agent_api.execute_deep_research_agent(request=request),
      media_type="text/event-stream",
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  fast_api_app.include_router(router)
  # Run the FastAPI application using uvicorn
  uvicorn.run(fast_api_app, host=_HOST.value, port=_PORT.value)


if __name__ == "__main__":
  app.run(main)
