"""Creates a node agent that accepts an objective and iterates until success or failure."""

import logging
from google.adk.agents import llm_agent
from google.adk.planners import built_in_planner
from google.genai import types
from opal_adk.types import models
from opal_adk.types import ui_type as opal_adk_ui_types
from opal_adk.util import tool_utils

UIType = opal_adk_ui_types.UIType

AGENT_NAME = "opal_adk_node_agent"
OUTPUT_KEY = "opal_adk_node_agent_output"
_TEMPERATURE = 1.0
_TOP_P = 1


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
  instructions_and_tools = tool_utils.get_tools()
  ui_type_instructions_and_tools = tool_utils.get_tools_for_ui_type(
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
