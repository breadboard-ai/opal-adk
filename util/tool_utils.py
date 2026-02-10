"""Utility functions for managing tools and agent functions."""

from collections.abc import Callable
from typing import Any, List, Tuple
from opal_adk.tools.chat import chat_request_user_input
from opal_adk.tools.chat import instructions as chat_instructions
from opal_adk.tools.generate import generate_text
from opal_adk.tools.generate import instructions as generate_instructions
from opal_adk.tools.system import instructions as system_instructions
from opal_adk.tools.system import objective_failed
from opal_adk.tools.system import objective_fulfilled
from opal_adk.types import ui_type as opal_adk_ui_type

UIType = opal_adk_ui_type.UIType


def get_tools() -> List[Tuple[str, List[Callable[..., Any]]]]:
  """Returns all available instructions and tools.

  Args:

  Returns:
    A list of tuples, each containing system instructions (str) and a list of
    tool callables.
  """
  return [
      (
          system_instructions.SYSTEM_FUNCTIONS_INSTRUCTIONS,
          [
              objective_failed.objective_failed,
              objective_fulfilled.objective_fulfilled,
          ],
      ),
      (
          generate_instructions.GENERATE_INSTRUCTIONS,
          [generate_text.generate_text],
      ),
  ]


def get_tools_for_ui_type(
    ui_type: UIType,
) -> List[Tuple[str, List[Callable[..., Any]]]]:
  """Returns instructions and callables for tools based on the UI type.

  Args:
    ui_type: The UI type to filter tools by.

  Returns:
    A list of tuples, each containing system instructions (str) and a list of
    tool callables.
  """

  match ui_type:
    case UIType.CHAT:
      return [(
          chat_instructions.CHAT_INSTRUCTIONS,
          [chat_request_user_input.chat_request_user_input],
      )]
    case UIType.A2UI:
      raise NotImplementedError(
          f"tools_utils: UI type {ui_type} is not yet implemented."
      )
    case _:
      return []
