"""Tests for vertex_generate_audio."""

from unittest import mock

from absl.testing import absltest
from opal_adk.error_handling import opal_adk_error
from opal_adk.tools.generate.generate_utils import vertex_generate_audio
from opal_adk.types import models


class VertexGenerateAudioTest(absltest.TestCase):

  @mock.patch.object(
      vertex_generate_audio.vertex_ai_client,
      'create_vertex_ai_client',
      autospec=True,
  )
  def test_generate_audio_success(self, mock_create_client):
    mock_client = mock.Mock()
    mock_create_client.return_value = mock_client

    mock_response = mock.Mock()
    mock_candidate = mock.Mock()
    mock_content = mock.Mock()
    mock_part = mock.Mock()

    mock_part.inline_data.data = b'fake_pcm_data'
    mock_content.parts = [mock_part]
    mock_candidate.content = mock_content
    mock_response.candidates = [mock_candidate]

    mock_client.models.generate_content.return_value = mock_response

    audio_bytes, mime_type = vertex_generate_audio.generate_audio(
        'test text', voice_name='Kore'
    )

    self.assertTrue(audio_bytes.startswith(b'RIFF'))
    self.assertEqual(mime_type, 'audio/wav')

    mock_client.models.generate_content.assert_called_once()
    _, kwargs = mock_client.models.generate_content.call_args
    self.assertEqual(kwargs['model'], models.Models.GEMINI_2_5_FLASH_TTS.value)
    self.assertEqual(kwargs['contents'].parts[0].text, 'test text')
    self.assertEqual(kwargs['contents'].role, 'user')

    config = kwargs['config']
    self.assertEqual(config.response_modalities, ['AUDIO'])
    self.assertEqual(
        config.speech_config.voice_config.prebuilt_voice_config.voice_name,
        'Kore',
    )

  @mock.patch.object(
      vertex_generate_audio.vertex_ai_client,
      'create_vertex_ai_client',
      autospec=True,
  )
  def test_generate_audio_truncates_long_text(self, mock_create_client):
    mock_client = mock.Mock()
    mock_create_client.return_value = mock_client

    mock_response = mock.Mock()
    mock_candidate = mock.Mock()
    mock_content = mock.Mock()
    mock_part = mock.Mock()

    mock_part.inline_data.data = b'fake_pcm_data'
    mock_content.parts = [mock_part]
    mock_candidate.content = mock_content
    mock_response.candidates = [mock_candidate]
    mock_client.models.generate_content.return_value = mock_response

    long_text = 'a' * 2000
    vertex_generate_audio.generate_audio(long_text)

    _, kwargs = mock_client.models.generate_content.call_args
    sent_text = kwargs['contents'].parts[0].text
    self.assertLen(sent_text, 1000)

  @mock.patch.object(
      vertex_generate_audio.vertex_ai_client,
      'create_vertex_ai_client',
      autospec=True,
  )
  def test_generate_audio_no_candidates(self, mock_create_client):
    mock_client = mock.Mock()
    mock_create_client.return_value = mock_client

    mock_response = mock.Mock()
    mock_response.candidates = []

    mock_client.models.generate_content.return_value = mock_response

    with self.assertRaisesRegex(
        opal_adk_error.OpalAdkError, 'No audio generated'
    ):
      vertex_generate_audio.generate_audio('test text')

  @mock.patch.object(
      vertex_generate_audio.vertex_ai_client,
      'create_vertex_ai_client',
      autospec=True,
  )
  def test_generate_audio_api_error(self, mock_create_client):
    mock_client = mock.Mock()
    mock_create_client.return_value = mock_client

    mock_client.models.generate_content.side_effect = Exception('API failed')

    with self.assertRaisesRegex(
        opal_adk_error.OpalAdkError, 'Failed to generate audio'
    ):
      vertex_generate_audio.generate_audio('test text')


if __name__ == '__main__':
  absltest.main()
