"""Tests for opal_adk.tools.vertex_search."""

import unittest
from unittest import mock

from google import genai
from google.genai import types
from opal_adk.constants import models
from opal_adk.tools import vertex_search


class VertexSearchTest(unittest.TestCase):

  @mock.patch('opal_adk.util.gemini_utils.extract_grounding_metadata')
  def test_search_via_vertex_ai_successful(self, mock_extract_metadata):
    """Tests a successful search that returns grounding metadata."""

    mock_genai_client = mock.MagicMock(spec=genai.Client)
    mock_response = mock.MagicMock(spec=types.GenerateContentResponse)
    mock_response.text = 'Test search result'
    mock_response.candidates = [mock.MagicMock()]  # Has candidates
    mock_genai_client.models.generate_content.return_value = mock_response

    expected_metadata = [types.Content()]
    mock_extract_metadata.return_value = expected_metadata

    result, metadata = vertex_search.search_via_vertex_ai(
        genai_client=mock_genai_client, query='test query'
    )

    self.assertEqual(result, 'Test search result')
    self.assertEqual(metadata, expected_metadata)
    mock_genai_client.models.generate_content.assert_called_once()
    mock_extract_metadata.assert_called_once_with(mock_response)

  def test_search_via_vertex_ai_no_candidates(self):
    """Tests a search that returns no candidates."""

    mock_genai_client = mock.MagicMock(spec=genai.Client)
    mock_response = mock.MagicMock(spec=types.GenerateContentResponse)
    mock_response.text = 'Test search result'
    mock_response.candidates = []  # No candidates
    mock_genai_client.models.generate_content.return_value = mock_response

    with mock.patch(
        'opal_adk.util.gemini_utils.extract_grounding_metadata'
    ) as mock_extract_metadata:
      result, metadata = vertex_search.search_via_vertex_ai(
          genai_client=mock_genai_client, query='test query'
      )

      self.assertEqual(result, 'Test search result')
      self.assertEqual(metadata, [])
      mock_genai_client.models.generate_content.assert_called_once()
      mock_extract_metadata.assert_not_called()

  def test_common_vertex_search(self):
    """Tests that _common_vertex_search calls the search_fn correctly."""
    mock_search_fn = mock.MagicMock()
    query = 'test query'
    text_safety_settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        )
    ]

    vertex_search._common_vertex_search(
        mock_search_fn, query, text_safety_settings
    )

    mock_search_fn.assert_called_once()
    _, kwargs = mock_search_fn.call_args
    self.assertEqual(
        kwargs['model'], models.Models.MODEL_GEMINI_2_5_FLASH.value
    )
    self.assertEqual(kwargs['contents'], query)
    config = kwargs['config']
    self.assertIsInstance(config.tools[0].google_search, types.GoogleSearch)
    self.assertEqual(config.safety_settings, text_safety_settings)

  def test_common_vertex_search_returns_iterator(self):
    """Tests that _common_vertex_search returns an iterator."""
    # Non-streaming case
    mock_search_fn_non_stream = mock.MagicMock()
    mock_response = mock.MagicMock(spec=types.GenerateContentResponse)
    mock_response.text = 'Test search result'
    mock_search_fn_non_stream.return_value = mock_response
    result_iterator = vertex_search._common_vertex_search(
        mock_search_fn_non_stream, 'q'
    )
    self.assertEqual(list(result_iterator), [mock_response])

    # Streaming case
    mock_search_fn_stream = mock.MagicMock()
    mock_stream_response_part = mock.MagicMock(
        spec=types.GenerateContentResponse
    )
    mock_stream_response_part.text = 'Test search result part'
    mock_iterator = iter([mock_stream_response_part])
    mock_search_fn_stream.return_value = mock_iterator
    result_iterator = vertex_search._common_vertex_search(
        mock_search_fn_stream, 'q'
    )
    self.assertIs(result_iterator, mock_iterator)

  @mock.patch('opal_adk.tools.vertex_search._common_vertex_search')
  def test_search_via_vertex_ai_stream(self, mock_common_search):
    """Tests the streaming search function."""
    mock_genai_client = mock.MagicMock(spec=genai.Client)
    query = 'test query'
    text_safety_settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        )
    ]
    mock_stream_response_part = mock.MagicMock(
        spec=types.GenerateContentResponse
    )
    mock_stream_response_part.text = 'Test search result part'
    expected_result = iter([mock_stream_response_part])
    mock_common_search.return_value = expected_result

    result = vertex_search.search_via_vertex_ai_stream(
        mock_genai_client, query, text_safety_settings
    )

    self.assertEqual(result, expected_result)
    mock_common_search.assert_called_once_with(
        mock_genai_client.models.generate_content_stream,
        query,
        text_safety_settings,
    )


if __name__ == '__main__':
  unittest.main()
