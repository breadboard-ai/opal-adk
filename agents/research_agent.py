"""Agent for performing multi-step research using various tools."""

from collections.abc import Callable, Sequence
import logging
from typing import Any
from google.adk.agents import llm_agent
from google.adk.planners import built_in_planner
from google.genai import types
from opal_adk import models
from opal_adk.tools import fetch_url_contents
from opal_adk.tools import map_search
from opal_adk.tools import vertex_search


LlmAgent = llm_agent.LlmAgent


_DEFAULT_RESEARCH_TOOLS: Sequence[Callable[..., Any]] = (
    map_search.search_google_maps_places,
    fetch_url_contents.fetch_url,
    vertex_search.search_via_vertex_ai,
)

AGENT_NAME = 'opal_adk_research_agent'
OUTPUT_KEY = 'opal_adk_research_agent_output'


def research_system_instructions(is_first: bool) -> str:
  which = 'first' if is_first else 'next'
  return f"""
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
    """


def deep_research_agent(
    user_query: str,
    *,
    model: models.Models = models.Models.GEMINI_2_5_FLASH,
    is_first_iteration: bool = True,
    additional_tools: Sequence[Callable[..., Any]] | None = None,
) -> LlmAgent:
  """Creates an LlmAgent configured for research.

  This agent uses a set of default research tools and can be extended with
  additional tools. It's designed to perform multi-step research based on a
  user query.

  Args:
    user_query: The initial query from the user to start the research.
    model: The model to use with the agent. Defaults to Gemini 2.5 Flash.
    is_first_iteration: True if this is the first iteration of a multi-iteration
      agent run.
    additional_tools: An optional sequence of additional callable tools to
      include in the agent's tool set.

  Returns:
    An instance of base_agent.BaseAgent (specifically, an LlmAgent) configured
    for research.

  Raises:
    ValueError: If the user_query is empty or missing.
  """
  all_research_tools = list(_DEFAULT_RESEARCH_TOOLS)
  if not user_query:
    logging.error(
        'research_agent: User query is missing or empty, received: %s',
        user_query,
    )
    raise ValueError('research_agent: User query is missing or empty.')

  if additional_tools:
    all_research_tools.extend(additional_tools)
  thinking_config = types.ThinkingConfig(include_thoughts=True)
  return LlmAgent(
      name=AGENT_NAME,
      model=model.value,
      description=(
          'Makes use of research tools, such as web search and url fetching to'
          ' perform research given a user query.'
      ),
      tools=all_research_tools,
      instruction=research_system_instructions(is_first_iteration),
      planner=built_in_planner.BuiltInPlanner(thinking_config=thinking_config),
      output_key=OUTPUT_KEY,
  )
