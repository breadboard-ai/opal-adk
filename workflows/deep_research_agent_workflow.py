"""Workflow for a deep research agent that generates cited reports.

This file defines a sequential agent workflow that first performs in-depth
research using a `research_agent` and then compiles the findings into a
cited report using a `report_writing_agent`.
"""

from google.adk.agents import sequential_agent
from opal_adk.agents import report_writing_agent
from opal_adk.agents import research_agent
from opal_adk.types import models

_RESEARCH_AGENT_MODEL = models.Models.GEMINI_2_5_FLASH
_REPORT_WRITING_AGENT_MODEL = models.Models.GEMINI_2_5_FLASH
WORKFLOW_NAME = "opal_adk_deep_research_workflow"


def deep_research_agent_workflow(num_iterations: int = 5):
  """Creates a sequential agent workflow for deep research and report writing.

  This workflow combines a research agent and a report writing agent to first
  perform in-depth research based on a user query and then generate a cited
  research report.

  Args:
      num_iterations: The number of iterations the research agent should
        perform.

  Returns:
      A `sequential_agent.SequentialAgent` instance configured for deep
      research.
  """
  return sequential_agent.SequentialAgent(
      name=WORKFLOW_NAME,
      description=(
          "A multi-agent sequential flow that performs research based on a user"
          " query and then generates a cited research report."
      ),
      sub_agents=[
          research_agent.deep_research_agent(
              model=_RESEARCH_AGENT_MODEL, iterations=num_iterations
          ),
          report_writing_agent.report_writing_agent(
              model=_REPORT_WRITING_AGENT_MODEL,
              parent_agent_output_key=research_agent.OUTPUT_KEY,
          ),
      ],
  )
