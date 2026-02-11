"""Utilities for handling Gemini API responses."""

import logging
from google.genai import types

Part = types.Part


def extract_grounding_metadata(
    response: types.GenerateContentResponse,
) -> list[types.Content]:
  """Extracts grounding metadata from Vertex AI generate_content response."""
  result = []
  candidate = response.candidates[0]
  if not candidate.grounding_metadata:
    return result
  logging.info('Grounding metadata: %s', candidate.grounding_metadata)
  if candidate.grounding_metadata.web_search_queries:
    result.append(
        types.Content(
            parts=[
                Part(
                    text='\n\n\nRelated Google Search queries: '
                    + ', '.join(
                        candidate.grounding_metadata.web_search_queries
                    ),
                )
            ]
        )
    )
  if candidate.grounding_metadata.grounding_chunks:
    grounding_chunks = '\n\nSources:\n'
    for grounding_chunk in candidate.grounding_metadata.grounding_chunks:
      if not grounding_chunk.web:
        logging.info('Unexpected grounding chunk: %s', grounding_chunk)
      else:
        grounding_chunks += (
            f'{grounding_chunk.web.title}: {grounding_chunk.web.uri}\n'
        )
    result.append(
        types.Content(
            parts=[Part(text=grounding_chunks)],
        )
    )
  return result
