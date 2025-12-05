"""Agent for performing multi-step research using various tools."""

from collections.abc import Callable, Sequence
import textwrap
from typing import Any
from google.adk.agents import llm_agent
from google.adk.planners import built_in_planner
from google.genai import types
from opal_adk import models
from opal_adk.clients import vertex_ai_client
from opal_adk.tools import fetch_url_contents
from opal_adk.tools import map_search_tool
from opal_adk.tools import vertex_search_tool


AGENT_NAME = 'opal_adk_research_agent'
OUTPUT_KEY = 'opal_adk_research_agent_output'


def research_system_instructions(is_first: bool) -> str:
  which = 'first' if is_first else 'next'
  return textwrap.dedent(f"""
    Your job is to use the provided query to produce raw research that will be later turned into a detailed research report.
    You are tasked with finding as much of relevant information as possible.

    You examine the conversation context so far and come up with the {which} step to produce the report,
    using the conversation context as the the guide of steps taken so far and the outcomes recorded.

    You do not ask user for feedback. You do not try to have a conversation with the user.
    You know that the user will only ask you to proceed to next step.

    Looking back at all that you've researched and the query/research plan, do you have enough to produce the detailed report? If so, you are done.

    Now, provide a response. Your response must contain two parts:
    Thought: a brief plain text reasoning why this is the right {which} step and a description of what you will do in plain English.
    Action: invoking the tools are your disposal, more than one if necessary. If you're done, do not invoke any tools.
    """)


def previous_agent_output_instructions(output_key: str) -> str:
  output_key = f'{output_key}'
  return textwrap.dedent(f"""
    You should make use of the additional information provided in the following:
    
    {output_key}
  """)


def deep_research_agent(
    *,
    parent_agent_output_key: str | None = None,
    model: models.Models = models.Models.GEMINI_2_5_FLASH,
    is_first_iteration: bool = True,
    additional_tools: Sequence[Callable[..., Any]] | None = None,
) -> llm_agent.LlmAgent:
  """Creates an LlmAgent configured for research.

  This agent uses a set of default research tools and can be extended with
  additional tools. It's designed to perform multi-step research based on a
  user query.

  Args:
    parent_agent_output_key: The output key of the previous agent. This agent
      has output to consider.
    model: The model to use with the agent. Defaults to Gemini 2.5 Flash.
    is_first_iteration: True if this is the first iteration of a multi-iteration
      agent run.
    additional_tools: An optional sequence of additional callable tools to
      include in the agent's tool set.

  Returns:
    An instance of base_agent.BaseAgent (specifically, an LlmAgent) configured
    for research.
  """
  all_research_tools = [
      map_search_tool.MapSearchTool(),
      vertex_search_tool.VertexSearchTool(
          genai_client=vertex_ai_client.create_vertex_ai_client()
      ),
      fetch_url_contents.fetch_url,
  ]
  if additional_tools:
    all_research_tools.extend(additional_tools)

  agent_instructions = research_system_instructions(is_first_iteration)
  if parent_agent_output_key:
    agent_instructions += previous_agent_output_instructions(
        parent_agent_output_key
    )

  thinking_config = types.ThinkingConfig(include_thoughts=True)
  return llm_agent.LlmAgent(
      name=AGENT_NAME,
      model=model.value,
      description=(
          'Makes use of research tools, such as web search and url fetching to'
          ' perform research given a user query.'
      ),
      tools=all_research_tools,
      instruction=agent_instructions,
      planner=built_in_planner.BuiltInPlanner(thinking_config=thinking_config),
      output_key=OUTPUT_KEY,
  )
