"""ADK compliant tool functions for performing searches via Vertex AI.

This module provides functionality to leverage Google Search through the Vertex
AI `generate_content` API, specifically using the `google_search` tool.
"""

from typing import Any, Sequence

from google import genai
from google.adk.tools import base_tool
from google.adk.tools import tool_context
from google.genai import types
from opal_adk.constants import models
from opal_adk.util import gemini_utils


_MAX_OUTPUT_TOKENS = 20000
_TEMPERATURE = 0.0
_THINKING_BUDGET = 0


Models = models.Models


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
      model: str = Models.MODEL_GEMINI_2_5_FLASH.value,
  ):
    super().__init__(
        name="OpalAdkVertexSearchTool",
        description="Performs a web search using Vertex AI.",
    )
    self.genai_client = genai_client
    self.text_safety_settings = text_safety_settings
    self.model = model

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
