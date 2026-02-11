from unittest import mock

from absl.testing import absltest
from google.api_core import exceptions as api_core_exceptions
from opal_adk.error_handling import opal_adk_error
from opal_adk.tools.generate.generate_utils import vertex_generate_video


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

    video_bytes, mime_type = vertex_generate_video.generate_video_via_vertex_ai(
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
    mock_error = api_core_exceptions.ResourceExhausted('error')

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

    video_bytes, mime_type = vertex_generate_video.generate_video_via_vertex_ai(
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
    mock_error = api_core_exceptions.ResourceExhausted('error')
    mock_client.models.generate_videos.side_effect = mock_error

    with self.assertRaises(opal_adk_error.OpalAdkError):
      vertex_generate_video.generate_video_via_vertex_ai('test video prompt')

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

    with self.assertRaisesRegex(opal_adk_error.OpalAdkError, 'bad prompt'):
      vertex_generate_video.generate_video_via_vertex_ai('test video prompt')
