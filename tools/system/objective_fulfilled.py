"""Defines a function that terminates a loop agent with a successful result."""

import logging
from google.adk.tools import tool_context as tc


def objective_fulfilled(
    tool_context: tc.ToolContext,
    response: str,
) -> dict[str, str]:
  """Indicates completion of the overall objective.

  Call only when the specified objective is entirely fulfilled.

  Args:
    tool_context: The context object containing information about the current
      tool execution.
    response: The final result to pass back to the user.

  Returns:
    A dictionary indicating the status of the operation.
  """
  tool_context.actions.escalate = True
  tool_context.actions.skip_summarization = True
  logging.info("SUCCESS! Objective fulfilled")
  return {"status": "success", "response": response}
