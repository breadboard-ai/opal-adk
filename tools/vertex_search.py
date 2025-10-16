"""ADK compliant tool functions for performing searches via Vertex AI.

This module provides functionality to leverage Google Search through the Vertex
AI `generate_content` API, specifically using the `google_search` tool.
"""

from google import genai
from google.genai import types
from opal_adk.constants import models
from opal_adk.util import gemini_utils

_MAX_OUTPUT_TOKENS = 20000
_TEMPERATURE = 0.0
_THINKING_BUDGET = 0


Models = models.Models


def search_via_vertex_ai(
    genai_client: genai.Client,
    query: str,
    text_safety_settings: list[types.SafetySetting] | None = None,
) -> tuple[str, list[types.Content]]:
  """Performs a web search with Cloud Vertex AI."""
  res = genai_client.models.generate_content(
      model=Models.MODEL_GEMINI_2_5_FLASH.value,
      contents=query,
      config=types.GenerateContentConfig(
          temperature=_TEMPERATURE,  # Recommended default for Search Grounding.
          max_output_tokens=_MAX_OUTPUT_TOKENS,
          thinking_config=types.ThinkingConfig(
              thinking_budget=_THINKING_BUDGET
          ),
          tools=[types.Tool(google_search=types.GoogleSearch())],
          safety_settings=text_safety_settings,
      ),
  )
  result = res.text
  if not res.candidates:
    return result, []
  grounding_metadata = gemini_utils.extract_grounding_metadata(res)
  return result, grounding_metadata
