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

class GenerateVideoViaVertexAiTest(absltest.TestCase):
  """Tests for generate_video_via_vertex_ai."""

  @mock.patch(
      'opal_adk.util.gemini_utils.vertex_ai_client.create_vertex_ai_client',
      autospec=True,
  )
  @mock.patch('time.sleep', autospec=True)
  def test_success(self, _, mock_create_client):
    mock_client = mock.Mock()
    mock_create_client.return_value = mock_client

    # Mock operation polling
    mock_operation = mock.Mock()
    mock_operation.done = False

    mock_done_operation = mock.Mock()
    mock_done_operation.done = True
    mock_done_operation.error = None

    mock_video = mock.Mock()
    mock_video.video.video_bytes = b'fake_video_bytes'
    mock_video.video.mime_type = 'video/mp4'
    mock_done_operation.response.generated_videos = [mock_video]

    mock_client.models.generate_videos.return_value = mock_operation
    mock_client.operations.get.return_value = mock_done_operation

    video_bytes, mime_type = gemini_utils.generate_video_via_vertex_ai(
        'test video prompt'
    )

    self.assertEqual(video_bytes, b'fake_video_bytes')
    self.assertEqual(mime_type, 'video/mp4')
    mock_client.models.generate_videos.assert_called_once()
    mock_client.operations.get.assert_called_once_with(mock_operation)

  @mock.patch(
      'opal_adk.util.gemini_utils.vertex_ai_client.create_vertex_ai_client',
      autospec=True,
  )
  def test_fallback_on_resource_exhausted(self, mock_create_client):
    mock_client = mock.Mock()
    mock_create_client.return_value = mock_client

    # First call raises an error, second succeeds
    mock_error = gemini_utils.api_core_exceptions.ResourceExhausted('error')

    mock_operation = mock.Mock()
    mock_operation.done = True
    mock_operation.error = None

    mock_video = mock.Mock()
    mock_video.video.video_bytes = b'fake_video_bytes2'
    mock_video.video.mime_type = 'video/mp4'
    mock_operation.response.generated_videos = [mock_video]

    mock_client.models.generate_videos.side_effect = [
        mock_error,
        mock_operation,
    ]

    video_bytes, mime_type = gemini_utils.generate_video_via_vertex_ai(
        'test video prompt',
    )

    self.assertEqual(video_bytes, b'fake_video_bytes2')
    self.assertEqual(mime_type, 'video/mp4')
    self.assertEqual(mock_client.models.generate_videos.call_count, 2)

  @mock.patch(
      'opal_adk.util.gemini_utils.vertex_ai_client.create_vertex_ai_client',
      autospec=True,
  )
  def test_exhausts_fallbacks(self, mock_create_client):
    mock_client = mock.Mock()
    mock_create_client.return_value = mock_client

    # All calls raise an error
    mock_error = gemini_utils.api_core_exceptions.ResourceExhausted('error')
    mock_client.models.generate_videos.side_effect = mock_error

    with self.assertRaises(gemini_utils.opal_adk_error.OpalAdkError):
      gemini_utils.generate_video_via_vertex_ai('test video prompt')

  @mock.patch(
      'opal_adk.util.gemini_utils.vertex_ai_client.create_vertex_ai_client',
      autospec=True,
  )
  def test_operation_error_raises(self, mock_create_client):
    mock_client = mock.Mock()
    mock_create_client.return_value = mock_client

    mock_operation = mock.Mock()
    mock_operation.done = True
    mock_operation.error = {'code': 400, 'message': 'bad prompt'}
    mock_operation.response = None

    mock_client.models.generate_videos.return_value = mock_operation

    with self.assertRaisesRegex(
        gemini_utils.opal_adk_error.OpalAdkError, 'bad prompt'
    ):
      gemini_utils.generate_video_via_vertex_ai('test video prompt')


if __name__ == '__main__':
  absltest.main()
