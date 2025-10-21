"""ADK compliant tool functions for performing searches via Vertex AI.

This module provides functionality to leverage Google Search through the Vertex
AI `generate_content` API, specifically using the `google_search` tool.
"""

from typing import Any, Callable, Iterator, Sequence
from google import genai
from google.genai import types
from opal_adk.constants import models
from opal_adk.util import gemini_utils

_MAX_OUTPUT_TOKENS = 20000
_TEMPERATURE = 0.0
_THINKING_BUDGET = 0


Models = models.Models


def _common_vertex_search(
    search_fn: Callable[..., Any],
    query: str,
    text_safety_settings: Sequence[types.SafetySetting] | None = None,
) -> Iterator[types.GenerateContentResponse]:
  """Performs a search using a Vertex AI generate_content function.

  This function is a common helper for both streaming and non-streaming
  Vertex AI search calls. It configures the call to use the `google_search`
  tool with predefined settings.

  Args:
    search_fn: The specific `generate_content` function to call
      (e.g., `genai_client.models.generate_content` or
      `genai_client.models.generate_content_stream`).
    query: The search query string.
    text_safety_settings: Optional safety settings to apply to the request.

  Returns:
    An iterator of `types.GenerateContentResponse` objects.
  """
  result = search_fn(
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

  if not isinstance(result, Iterator):
    return iter([result])
  return result


def search_via_vertex_ai_stream(
    genai_client: genai.Client,
    query: str,
    text_safety_settings: Sequence[types.SafetySetting] | None = None,
) -> Iterator[types.GenerateContentResponse]:
  """Generates text with Cloud Vertex AI in chunks."""
  return _common_vertex_search(
      genai_client.models.generate_content_stream, query, text_safety_settings
  )


def search_via_vertex_ai(
    genai_client: genai.Client,
    query: str,
    text_safety_settings: Sequence[types.SafetySetting] | None = None,
) -> tuple[str, Sequence[types.Content]]:
  """Performs a web search with Cloud Vertex AI."""
  res = next(_common_vertex_search(
      genai_client.models.generate_content, query, text_safety_settings
  ))

  result = res.text
  if not res.candidates:
    return result, []
  grounding_metadata = gemini_utils.extract_grounding_metadata(res)
  return result, grounding_metadata
