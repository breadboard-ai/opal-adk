"""Creates a node agent that accepts an objective and iterates until success or failure."""

import logging
from google.adk.agents import llm_agent
from google.adk.planners import built_in_planner
from google.genai import types
from opal_adk.types import model_constraint
from opal_adk.types import models
from opal_adk.util import tool_utils

AGENT_NAME = "opal_adk_node_agent"
OUTPUT_KEY = "opal_adk_node_agent_output"


def node_agent(
    constraint: model_constraint.ModelConstraint,
    agent_model: models.Models = models.Models.GEMINI_2_5_FLASH,
) -> llm_agent.LlmAgent:
  """Creates an LlmAgent configured for writing research reports.

  This agent is designed to take raw research input and generate a thorough,
  detailed, and well-cited research report in markdown format.

  Args:
    constraint: The model constraint that will determine the tools available to
      the agent.
    agent_model: The model to use for the LLM agent. Defaults to
      GEMINI_2_5_FLASH.

  Returns:
    An instance of llm_agent.LlmAgent.
  """
  thinking_config = types.ThinkingConfig(include_thoughts=True)
  agent_instructions = []
  tools = []
  instructions_and_tools = tool_utils.get_tools_with_model_constraints(
      constraint
  )
  logging.info("node_agent: instructions_and_tools %s", instructions_and_tools)
  for agent_instruction, tool_list in instructions_and_tools:
    agent_instructions.append(agent_instruction)
    tools = tools + tool_list
  instructions = "\n".join(agent_instructions)
  return llm_agent.LlmAgent(
      name=AGENT_NAME,
      description="Iteratively works to solve the stated objective",
      model=agent_model.value,
      instruction=instructions,
      output_key=OUTPUT_KEY,
      planner=built_in_planner.BuiltInPlanner(thinking_config=thinking_config),
      tools=tools,
  )
