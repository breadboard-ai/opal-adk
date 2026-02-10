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


class GenerateImageViaGenaiApiTest(absltest.TestCase):
  """Tests for generate_image_via_genai_api."""

  @mock.patch(
      'opal_adk.util.gemini_utils.vertex_ai_client.create_vertex_ai_client'
  )
  def test_success_first_model(self, mock_create_client):
    mock_client = mock.Mock()
    mock_create_client.return_value = mock_client

    # Mock successful image generation
    mock_image = mock.Mock()
    mock_image.image.image_bytes = b'fake_image_bytes'
    mock_image.image.mime_type = 'image/png'
    mock_client.models.generate_images.return_value = [mock_image]

    image_bytes, mime_type = gemini_utils.generate_image_via_genai_api(
        'test prompt'
    )

    self.assertEqual(image_bytes, b'fake_image_bytes')
    self.assertEqual(mime_type, 'image/png')
    self.assertEqual(mock_client.models.generate_images.call_count, 1)

  @mock.patch(
      'opal_adk.util.gemini_utils.vertex_ai_client.create_vertex_ai_client'
  )
  def test_fallback_on_retriable_error(self, mock_create_client):
    mock_client = mock.Mock()
    mock_create_client.return_value = mock_client

    # First call raises an error, second succeeds
    mock_error = gemini_utils.api_core_exceptions.ResourceExhausted('error')

    mock_image = mock.Mock()
    mock_image.image.image_bytes = b'fake_image_bytes2'
    mock_image.image.mime_type = 'image/jpeg'

    mock_client.models.generate_images.side_effect = [mock_error, [mock_image]]

    image_bytes, mime_type = gemini_utils.generate_image_via_genai_api(
        'test prompt'
    )

    self.assertEqual(image_bytes, b'fake_image_bytes2')
    self.assertEqual(mime_type, 'image/jpeg')
    self.assertEqual(mock_client.models.generate_images.call_count, 2)

  @mock.patch(
      'opal_adk.util.gemini_utils.vertex_ai_client.create_vertex_ai_client'
  )
  def test_exhausts_fallbacks(self, mock_create_client):
    mock_client = mock.Mock()
    mock_create_client.return_value = mock_client

    # All calls raise an error
    mock_error = gemini_utils.api_core_exceptions.ResourceExhausted('error')
    mock_client.models.generate_images.side_effect = mock_error

    with self.assertRaises(gemini_utils.opal_adk_error.OpalAdkError):
      gemini_utils.generate_image_via_genai_api('test prompt')


if __name__ == '__main__':
  absltest.main()
