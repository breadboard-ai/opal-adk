"""Tests for opal_adk.tools.vertex_search."""

import unittest
from unittest import mock

from google import genai
from google.genai import types
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


if __name__ == '__main__':
  unittest.main()