"""Tests for generate_image_via_gemini_api."""

from unittest import mock

from absl.testing import absltest
from google.genai import types
from opal_adk.error_handling import opal_adk_error
from opal_adk.tools.generate.generate_utils import gemini_generate_image
from opal_adk.types import image_types
from opal_adk.types import models


class GeminiGenerateImageTest(absltest.TestCase):

  @mock.patch.object(
      gemini_generate_image.vertex_ai_client,
      'create_vertex_ai_client',
      autospec=True,
  )
  def test_generate_image_success(self, mock_create_client):
    mock_client = mock.Mock()
    mock_create_client.return_value = mock_client

    mock_response = mock.Mock()
    mock_candidate = mock.Mock()
    mock_content = mock.Mock()

    mock_image_part = mock.Mock()
    mock_image_part.inline_data = mock.Mock()
    mock_image_part.inline_data.data = b'fake_image_bytes'
    mock_image_part.inline_data.mime_type = 'image/png'
    mock_image_part.text = None

    mock_text_part = mock.Mock()
    mock_text_part.inline_data = None
    mock_text_part.text = 'Some extra text'

    mock_content.parts = [mock_image_part, mock_text_part]
    mock_candidate.content = mock_content
    mock_response.candidates = [mock_candidate]

    mock_client.models.generate_content.return_value = mock_response

    parts = [types.Part(text='A test image')]
    images = gemini_generate_image.gemini_generate_images(
        parts,
        aspect_ratio=image_types.AspectRatio.RATIO_16_9,
    )

    self.assertLen(images, 1)
    self.assertEqual(images[0][0], b'fake_image_bytes')
    self.assertEqual(images[0][1], 'image/png')

    mock_client.models.generate_content.assert_called_once()
    _, kwargs = mock_client.models.generate_content.call_args
    self.assertEqual(
        kwargs['model'], models.Models.GEMINI_2_5_FLASH_IMAGE.value
    )
    self.assertEqual(kwargs['contents'].parts[0].text, 'A test image')
    self.assertEqual(kwargs['contents'].role, 'user')
    config = kwargs['config']
    self.assertEqual(config.response_modalities, ['TEXT', 'IMAGE'])
    self.assertEqual(config.image_config.aspect_ratio, '16:9')

  @mock.patch.object(
      gemini_generate_image.vertex_ai_client,
      'create_vertex_ai_client',
      autospec=True,
  )
  def test_generate_image_no_candidates(self, mock_create_client):
    mock_client = mock.Mock()
    mock_create_client.return_value = mock_client

    mock_response = mock.Mock()
    mock_response.candidates = []

    mock_client.models.generate_content.return_value = mock_response

    parts = [types.Part(text='A test image')]
    with self.assertRaisesRegex(
        opal_adk_error.OpalAdkError, 'No candidates returned'
    ):
      gemini_generate_image.gemini_generate_images(parts)

  @mock.patch.object(
      gemini_generate_image.vertex_ai_client,
      'create_vertex_ai_client',
      autospec=True,
  )
  def test_generate_image_api_error(self, mock_create_client):
    mock_client = mock.Mock()
    mock_create_client.return_value = mock_client

    mock_client.models.generate_content.side_effect = Exception('API failed')

    parts = [types.Part(text='A test image')]
    with self.assertRaisesRegex(opal_adk_error.OpalAdkError, 'API failed'):
      gemini_generate_image.gemini_generate_images(parts)

  @mock.patch.object(
      gemini_generate_image.vertex_ai_client,
      'create_vertex_ai_client',
      autospec=True,
  )
  def test_generate_image_text_only_response(self, mock_create_client):
    mock_client = mock.Mock()
    mock_create_client.return_value = mock_client

    mock_response = mock.Mock()
    mock_candidate = mock.Mock()
    mock_content = mock.Mock()

    mock_text_part = mock.Mock()
    mock_text_part.inline_data = None
    mock_text_part.text = 'I cannot generate an image.'

    mock_content.parts = [mock_text_part]
    mock_candidate.content = mock_content
    mock_response.candidates = [mock_candidate]

    mock_client.models.generate_content.return_value = mock_response

    parts = [types.Part(text='A test image')]
    with self.assertRaisesRegex(
        opal_adk_error.OpalAdkError, 'No images generated.'
    ):
      gemini_generate_image.gemini_generate_images(parts)

  @mock.patch.object(
      gemini_generate_image.vertex_ai_client,
      'create_vertex_ai_client',
      autospec=True,
  )
  def test_generate_image_no_content_parts(self, mock_create_client):
    mock_client = mock.Mock()
    mock_create_client.return_value = mock_client

    mock_response = mock.Mock()
    mock_candidate = mock.Mock()
    mock_content = mock.Mock()

    mock_content.parts = []
    mock_candidate.content = mock_content
    mock_response.candidates = [mock_candidate]

    mock_client.models.generate_content.return_value = mock_response

    parts = [types.Part(text='A test image')]
    with self.assertRaisesRegex(
        opal_adk_error.OpalAdkError, 'No content parts returned'
    ):
      gemini_generate_image.gemini_generate_images(parts)


if __name__ == '__main__':
  absltest.main()

  @mock.patch.object(
      gemini_generate_image.vertex_ai_client,
      'create_vertex_ai_client',
      autospec=True,
  )
  def test_generate_image_with_optional_params(self, mock_create_client):
    mock_client = mock.Mock()
    mock_create_client.return_value = mock_client

    mock_response = mock.Mock()
    mock_candidate = mock.Mock()
    mock_content = mock.Mock()

    mock_image_part = mock.Mock()
    mock_image_part.inline_data = mock.Mock()
    mock_image_part.inline_data.data = b'fake_image_bytes'
    mock_image_part.inline_data.mime_type = 'image/png'
    mock_image_part.text = None

    mock_content.parts = [mock_image_part]
    mock_candidate.content = mock_content
    mock_response.candidates = [mock_candidate]

    mock_client.models.generate_content.return_value = mock_response

    parts = [types.Part(text='A test image')]
    safety_settings = [types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    )]
    
    images = gemini_generate_image.gemini_generate_images(
        parts,
        aspect_ratio=image_types.AspectRatio.RATIO_4_3.value,
        model_name='custom_model',
        safety_settings=safety_settings,
    )

    self.assertLen(images, 1)

    mock_client.models.generate_content.assert_called_once()
    _, kwargs = mock_client.models.generate_content.call_args
    self.assertEqual(kwargs['model'], 'custom_model')
    config = kwargs['config']
    self.assertEqual(config.safety_settings, safety_settings)
    self.assertEqual(config.image_config.aspect_ratio, '4:3')
