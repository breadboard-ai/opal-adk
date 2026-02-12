"""Tests for generate_speech_from_text."""

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from google.genai import types
from opal_adk.error_handling import opal_adk_error
from opal_adk.tools.generate import generate_speech_from_text
from google.rpc import code_pb2

class GenerateSpeechFromTextTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='male_voice',
          voice_input='en_US_male',
          expected_voice_name='Charon',
      ),
      dict(
          testcase_name='female_voice',
          voice_input='en_US_female',
          expected_voice_name='Zephyr',
      ),
  )
  @mock.patch(
      'opal_adk.tools.generate.generate_utils.vertex_generate_audio.generate_audio',
      autospec=True,
  )
  def test_generate_speech_from_text_success(
      self, mock_generate_audio, voice_input, expected_voice_name
  ):
    mock_generate_audio.return_value = (b'audio_bytes', 'audio/wav')
    input_text = ['Hello', 'World']

    result = generate_speech_from_text.generate_speech_from_text(
        input_text, voice=voice_input
    )

    self.assertLen(result, 2)
    self.assertIsInstance(result[0], types.Content)
    self.assertIsInstance(result[1], types.Content)
    self.assertEqual(result[0].parts[0].inline_data.data, b'audio_bytes')
    self.assertEqual(result[0].parts[0].inline_data.mime_type, 'audio/wav')
    self.assertEqual(result[1].parts[0].inline_data.data, b'audio_bytes')

    self.assertEqual(mock_generate_audio.call_count, 2)
    mock_generate_audio.assert_any_call('Hello', expected_voice_name)
    mock_generate_audio.assert_any_call('World', expected_voice_name)

  def test_generate_speech_from_text_default_voice(self):
    with mock.patch(
        'opal_adk.tools.generate.generate_utils.vertex_generate_audio.generate_audio',
        autospec=True,
    ) as mock_generate_audio:
      mock_generate_audio.return_value = (b'bytes', 'audio/mp3')
      input_text = ['Test']

      result = generate_speech_from_text.generate_speech_from_text(input_text)

      self.assertLen(result, 1)
      mock_generate_audio.assert_called_once_with('Test', 'Zephyr')

  def test_generate_speech_from_text_invalid_voice(self):
    with self.assertRaises(opal_adk_error.OpalAdkError) as cm:
      generate_speech_from_text.generate_speech_from_text(
          ['Test'], voice='invalid_voice'
      )

    self.assertEqual(cm.exception.error_code, code_pb2.INVALID_ARGUMENT)
    self.assertIn('Received invalid voice option', cm.exception.status_message)

  @mock.patch(
      'opal_adk.tools.generate.generate_utils.vertex_generate_audio.generate_audio',
      autospec=True,
  )
  def test_generate_speech_from_text_internal_error(self, mock_generate_audio):
    mock_generate_audio.side_effect = Exception('Something went wrong')

    with self.assertRaises(opal_adk_error.OpalAdkError) as cm:
      generate_speech_from_text.generate_speech_from_text(['Test'])

    self.assertEqual(cm.exception.error_code, code_pb2.INTERNAL)
    self.assertEqual(
        cm.exception.status_message,
        opal_adk_error.MODEL_CALL_ERROR_MESSAGE,
    )
    self.assertIn('Something went wrong', cm.exception.internal_details)


if __name__ == '__main__':
  absltest.main()
