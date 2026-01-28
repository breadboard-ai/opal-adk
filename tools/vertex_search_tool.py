"""ADK compliant tool functions for performing searches via Vertex AI.

This module provides functionality to leverage Google Search through the Vertex
AI `generate_content` API, specifically using the `google_search` tool.
"""

from typing import Any, Sequence

from google import genai
from google.adk.agents import llm_agent
from google.adk.tools import agent_tool
from google.adk.tools import base_tool
from google.adk.tools import tool_context
from google.genai import types
from opal_adk.clients import vertex_ai_client
from opal_adk.types import models
from opal_adk.util import gemini_utils


_MAX_OUTPUT_TOKENS = 20000
_TEMPERATURE = 1.0
_THINKING_BUDGET = 0
_SEARCH_AGENT_TOOL_INSTRUCTIONS = (
    "You are a specialist in using Vertex AI search given a topic to search. "
    " You MUST use the VertexSearchTool provided to perform you search query."
)

Models = models.Models


def search_agent_tool(
    model: models.Models = models.Models.FLASH_MODEL_NAME,
) -> agent_tool.AgentTool:
  """Creates an AgentTool instance for performing Vertex AI searches.

  This agent tool wraps the `VertexSearchTool` to allow agents to utilize
  Google Search via Vertex AI's `generate_content` API, specifically
  addressing limitations where agents might not directly call other tools when
  Vertex or Google Search is involved.

  Args:
    model: The Vertex AI model to use for the search agent. Defaults to
      `models.Models.FLASH_MODEL_NAME`, which is a fast, light-weight model.

  Returns:
    An `agent_tool.AgentTool` configured to use `VertexSearchTool`.
  """
  return agent_tool.AgentTool(
      llm_agent.LlmAgent(
          model=model.value,
          name="opal_adk_vertex_search_agent_tool",
          description=(
              "Vertex search wrapped as an agent tool. This is to overcome the"
              " limitation that an agent cannot call any other tool if vertex"
              " or google search is used."
          ),
          instruction=_SEARCH_AGENT_TOOL_INSTRUCTIONS,
          tools=[
              VertexSearchTool(
                  model=model,
                  genai_client=vertex_ai_client.create_vertex_ai_client(),
              )
          ],
      )
  )


class VertexSearchTool(base_tool.BaseTool):
  """A tool for performing web searches using Vertex AI's google_search tool.

  This tool leverages the `generate_content` API with the `google_search`
  tool configured to provide search results. It can be used within an ADK
  framework to add web search capabilities.
  """

  def __init__(
      self,
      *,
      genai_client: genai.Client,
      text_safety_settings: Sequence[types.SafetySetting] | None = None,
      model: models.Models = models.Models.FLASH_MODEL_NAME,
  ):
    super().__init__(
        name="OpalAdkVertexSearchTool",
        description="Performs a web search using Vertex AI.",
    )
    self.genai_client = genai_client
    self.text_safety_settings = text_safety_settings
    self.model = model.value

  def __call__(
      self, query: str, context: tool_context.ToolContext
  ) -> dict[str, Any]:
    result = self.genai_client.models.generate_content(
        model=self.model,
        contents=query,
        config=types.GenerateContentConfig(
            temperature=_TEMPERATURE,
            max_output_tokens=_MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(
                thinking_budget=_THINKING_BUDGET
            ),
            tools=[types.Tool(google_search=types.GoogleSearch())],
            safety_settings=self.text_safety_settings,
        ),
    )

    if not result.candidates:
      return {"result": []}
    grounding_metadata = gemini_utils.extract_grounding_metadata(result)
    return {"result": result.text, "grounding_metadata:": grounding_metadata}
