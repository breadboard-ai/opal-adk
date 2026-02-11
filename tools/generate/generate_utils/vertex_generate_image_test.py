from unittest import mock

from absl.testing import absltest
from opal_adk.util import gemini_utils
from opal_adk.tools.generate.generate_utils import vertex_generate_image


class GenerateImageViaGenaiApiTest(absltest.TestCase):
  """Tests for generate_image_via_genai_api."""

  @mock.patch(
      'opal_adk.util.gemini_utils.vertex_ai_client.create_vertex_ai_client',
      autospec=True,
  )
  def test_success_first_model(self, mock_create_client):
    mock_client = mock.Mock()
    mock_create_client.return_value = mock_client

    # Mock successful image generation
    mock_image = mock.Mock()
    mock_image.image.image_bytes = b'fake_image_bytes'
    mock_image.image.mime_type = 'image/png'
    mock_client.models.generate_images.return_value = [mock_image]

    image_bytes, mime_type = vertex_generate_image.generate_image_via_genai_api(
        'test prompt'
    )

    self.assertEqual(image_bytes, b'fake_image_bytes')
    self.assertEqual(mime_type, 'image/png')
    self.assertEqual(mock_client.models.generate_images.call_count, 1)

  @mock.patch(
      'opal_adk.util.gemini_utils.vertex_ai_client.create_vertex_ai_client',
      autospec=True,
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

    image_bytes, mime_type = vertex_generate_image.generate_image_via_genai_api(
        'test prompt'
    )

    self.assertEqual(image_bytes, b'fake_image_bytes2')
    self.assertEqual(mime_type, 'image/jpeg')
    self.assertEqual(mock_client.models.generate_images.call_count, 2)

  @mock.patch(
      'opal_adk.util.gemini_utils.vertex_ai_client.create_vertex_ai_client',
      autospec=True,
  )
  def test_exhausts_fallbacks(self, mock_create_client):
    mock_client = mock.Mock()
    mock_create_client.return_value = mock_client

    # All calls raise an error
    mock_error = gemini_utils.api_core_exceptions.ResourceExhausted('error')
    mock_client.models.generate_images.side_effect = mock_error

    with self.assertRaises(gemini_utils.opal_adk_error.OpalAdkError):
      vertex_generate_image.generate_image_via_genai_api('test prompt')
