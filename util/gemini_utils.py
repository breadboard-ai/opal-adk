"""Utilities for handling Gemini API responses."""

import logging
from google.genai import types
from opal_adk.infra import environment_util
from opal_adk.error_handling import opal_adk_error
from google.rpc import code_pb2

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


def validate_candidate_recitation(
    response: types.GenerateContentResponse,
) -> None:
  """Checks if the candidate recitation is enabled."""
  for candidate in response.candidates:
    environment_util.log_if_not_staging(
        'Candidate finish reason: %s', candidate.finish_reason
    )
    if candidate.finish_reason == types.FinishReason.RECITATION:
      raise opal_adk_error.OpalAdkError(
          status_message=(
              'Recitation is found for the candidate. This may indicate a'
              ' policy violation in the generated text.'
          ),
          status_code=code_pb2.INTERNAL,
          details='Content blocked for recitation.',
      )
