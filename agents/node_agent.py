"""Creates a node agent that accepts an objective and iterates until success or failure."""

from collections.abc import Callable
import logging
from typing import Any, List, Tuple
from google.adk.agents import llm_agent
from google.adk.planners import built_in_planner
from google.genai import types
from opal_adk.tools.chat import chat_request_user_input
from opal_adk.tools.chat import instructions as chat_instructions
from opal_adk.tools.generate import generate_images
from opal_adk.tools.generate import generate_speech_from_text
from opal_adk.tools.generate import generate_text
from opal_adk.tools.generate import instructions as generate_instructions
from opal_adk.tools.system import instructions as system_instructions
from opal_adk.tools.system import objective_failed
from opal_adk.tools.system import objective_fulfilled
from opal_adk.types import models
from opal_adk.types import ui_type as opal_adk_ui_types

UIType = opal_adk_ui_types.UIType

AGENT_NAME = "opal_adk_node_agent"
OUTPUT_KEY = "opal_adk_node_agent_output"
_TEMPERATURE = 1.0
_TOP_P = 1


def _get_tools_for_ui_type(
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


def _get_tools() -> List[Tuple[str, List[Callable[..., Any]]]]:
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
          [
              generate_text.generate_text,
              generate_speech_from_text.generate_speech_from_text,
              generate_images.generate_images
          ],
      ),
  ]


def node_agent(
    ui_type: UIType,
    agent_model: models.Models = models.Models.GEMINI_3_FLASH,
) -> llm_agent.LlmAgent:
  """Creates an LlmAgent configured for writing research reports.

  This agent is designed to take raw research input and generate a thorough,
  detailed, and well-cited research report in markdown format.

  Args:
    ui_type: The UIType of the node. This can be either A2UI, CHAT or
      UNSPECIFIED. The type of UI will determine the tools available to the
      agent.
    agent_model: The model to use for the LLM agent. Defaults to
      GEMINI_2_5_FLASH.

  Returns:
    An instance of llm_agent.LlmAgent.
  """
  thinking_config = types.ThinkingConfig(include_thoughts=True)
  agent_instructions = []
  tools = []
  instructions_and_tools = _get_tools()
  ui_type_instructions_and_tools = _get_tools_for_ui_type(
      ui_type=ui_type
  )
  logging.debug("node_agent: instructions_and_tools %s", instructions_and_tools)
  for agent_instruction, tool_list in (
      instructions_and_tools + ui_type_instructions_and_tools
  ):
    agent_instructions.append(agent_instruction)
    tools.extend(tool_list)

  instructions = "\n".join(agent_instructions)
  config = types.GenerateContentConfig(
      temperature=1.0,
      top_p=1,
      tool_config={"function_calling_config": {"mode": "ANY"}},
  )
  return llm_agent.LlmAgent(
      name=AGENT_NAME,
      description="Iteratively works to solve the stated objective",
      model=agent_model.value,
      static_instruction=instructions,
      output_key=OUTPUT_KEY,
      generate_content_config=config,
      planner=built_in_planner.BuiltInPlanner(thinking_config=thinking_config),
      tools=tools,
  )
