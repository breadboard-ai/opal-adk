"""Defines a function that terminates a loop agent with a failed result."""

import logging
from google.adk.tools import tool_context as tc


def objective_failed(
    tool_context: tc.ToolContext,
) -> dict[str, str]:
  """Indicates that the agent failed to fulfill of the overall objective.

  Call ONLY when all means of fulfilling the objective have been exhausted.

  Args:
    tool_context: The context object containing information about the current
      tool execution.

  Returns:
    A dictionary indicating the status of the operation.
  """
  tool_context.actions.escalate = True
  tool_context.actions.skip_summarization = True
  logging.info("FAILED! Objective failed to be reached.")
  return {"status": "failed"}
