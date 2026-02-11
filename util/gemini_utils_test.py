from unittest import mock

from absl.testing import absltest
from google.genai import types
from opal_adk.util import gemini_utils


class ExtractGroundingMetadataTest(absltest.TestCase):
  """Tests for extract_grounding_metadata."""

  def test_no_grounding_metadata(self):
    response = mock.Mock(spec=types.GenerateContentResponse)
    response.candidates = [mock.Mock()]
    response.candidates[0].grounding_metadata = None
    self.assertEmpty(gemini_utils.extract_grounding_metadata(response))

  def test_empty_grounding_metadata(self):
    response = mock.Mock(spec=types.GenerateContentResponse)
    response.candidates = [mock.Mock()]
    response.candidates[0].grounding_metadata.web_search_queries = []
    response.candidates[0].grounding_metadata.grounding_chunks = []
    self.assertEmpty(gemini_utils.extract_grounding_metadata(response))

  def test_with_web_search_queries(self):
    response = mock.Mock(spec=types.GenerateContentResponse)
    response.candidates = [mock.Mock()]
    response.candidates[0].grounding_metadata.web_search_queries = [
        'query1',
        'query2',
    ]
    response.candidates[0].grounding_metadata.grounding_chunks = []

    result = gemini_utils.extract_grounding_metadata(response)

    expected = [
        types.Content(
            parts=[
                types.Part(
                    text='\n\n\nRelated Google Search queries: query1, query2'
                )
            ]
        )
    ]
    self.assertEqual(result, expected)

  def test_with_grounding_chunks(self):
    response = mock.Mock(spec=types.GenerateContentResponse)
    response.candidates = [mock.Mock()]
    response.candidates[0].grounding_metadata.web_search_queries = []
    chunk1 = mock.Mock()
    chunk1.web.title = 'title1'
    chunk1.web.uri = 'uri1'
    chunk2 = mock.Mock()
    chunk2.web.title = 'title2'
    chunk2.web.uri = 'uri2'
    response.candidates[0].grounding_metadata.grounding_chunks = [
        chunk1,
        chunk2,
    ]

    result = gemini_utils.extract_grounding_metadata(response)

    expected = [
        types.Content(
            parts=[
                types.Part(text='\n\nSources:\ntitle1: uri1\ntitle2: uri2\n')
            ]
        )
    ]
    self.assertEqual(result, expected)

  def test_with_mixed_grounding_chunks_ignores_none_web(self):
    response = mock.Mock(spec=types.GenerateContentResponse)
    response.candidates = [mock.Mock()]
    response.candidates[0].grounding_metadata.web_search_queries = []
    chunk1 = mock.Mock()
    chunk1.web.title = 'title1'
    chunk1.web.uri = 'uri1'
    chunk2 = mock.Mock()
    chunk2.web = None
    response.candidates[0].grounding_metadata.grounding_chunks = [
        chunk1,
        chunk2,
    ]

    result = gemini_utils.extract_grounding_metadata(response)

    expected = [
        types.Content(parts=[types.Part(text='\n\nSources:\ntitle1: uri1\n')])
    ]
    self.assertEqual(result, expected)

  def test_with_all_metadata(self):
    response = mock.Mock(spec=types.GenerateContentResponse)
    response.candidates = [mock.Mock()]
    response.candidates[0].grounding_metadata.web_search_queries = ['query1']
    chunk1 = mock.Mock()
    chunk1.web.title = 'title1'
    chunk1.web.uri = 'uri1'
    response.candidates[0].grounding_metadata.grounding_chunks = [chunk1]

    result = gemini_utils.extract_grounding_metadata(response)

    expected = [
        types.Content(
            parts=[
                types.Part(text='\n\n\nRelated Google Search queries: query1')
            ]
        ),
        types.Content(parts=[types.Part(text='\n\nSources:\ntitle1: uri1\n')]),
    ]
    self.assertEqual(result, expected)

