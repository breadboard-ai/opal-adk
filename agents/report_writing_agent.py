"""Creates a report writing agent that produces a cited report out of input."""

from google.adk.agents import llm_agent
from google.adk.planners import built_in_planner
from google.genai import types
from opal_adk import models

REPORT_WRITING_SYSTEM_INSTRUCTIONS = """
  You are a research report writer.
  Your teammates produced a wealth of raw research about the supplied query.

  Your task is to take the raw research and write a thorough, detailed research
  report that answers the provided query. Use markdown.

  A report must additionally contain references to the source
  (always cite your sources).

  When your report is completed, go back and judge your report and ensure you
  have fully answered the user query and
  all sources have been appropriately cited.
"""


def previous_agent_output_instructions(output_key: str):
  output_key = "{" + output_key + "}"
  return f"""
  You are provided with the following information to produce your report:

  {output_key}
"""


AGENT_NAME = "opal_adk_report_writing_agent"
OUTPUT_KEY = "opal_adk_report_writing_agent_output"


def report_writing_agent(
    model: models.Models = models.Models.GEMINI_2_5_FLASH,
    parent_agent_output_key: str | None = None,
):
  """Creates an LlmAgent configured for writing research reports.

  This agent is designed to take raw research input and generate a thorough,
  detailed, and well-cited research report in markdown format.

  Args:
    model: The model to use for the LLM agent. Defaults to GEMINI_2_5_FLASH.
    parent_agent_output_key: The output key of the parent agent.

  Returns:
    An instance of llm_agent.LlmAgent.
  """
  agent_instructions = REPORT_WRITING_SYSTEM_INSTRUCTIONS
  if parent_agent_output_key:
    agent_instructions += previous_agent_output_instructions(
        parent_agent_output_key
    )
  thinking_config = types.ThinkingConfig(include_thoughts=True)
  return llm_agent.LlmAgent(
      name=AGENT_NAME,
      description="Agent that takes input and produces a well cited report.",
      model=model.value,
      instruction=agent_instructions,
      output_key=OUTPUT_KEY,
      planner=built_in_planner.BuiltInPlanner(thinking_config=thinking_config),
  )
