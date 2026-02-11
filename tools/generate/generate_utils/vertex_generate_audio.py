"""Utility function to generate audio using Gemini TTS API."""

import io
import logging
import wave

from google.genai import types
from opal_adk.clients import vertex_ai_client
from opal_adk.error_handling import opal_adk_error
from opal_adk.types import models
from opal_adk.util import llm_logging

from google.rpc import code_pb2


def generate_audio(
    text: str,
    voice_name: str = 'Kore',
) -> tuple[bytes, str]:
  """Generates audio with Gemini TTS via GenAI API.

  Args:
    text: The text to convert to speech.
    voice_name: The voice to use for speech synthesis.

  Returns:
    Tuple of (audio_bytes, mime_type).
  """
  llm_logging.log_operation_start('Generate Audio (TTS)')
  logging.info('TTS text (truncated): %s', text[:1000])
  gemini_client = vertex_ai_client.create_vertex_ai_client()

  # Truncate to 1000 chars to avoid TTS API limits
  text = text[:1000]

  config = types.GenerateContentConfig(
      response_modalities=['AUDIO'],
      speech_config=types.SpeechConfig(
          voice_config=types.VoiceConfig(
              prebuilt_voice_config=types.PrebuiltVoiceConfig(
                  voice_name=voice_name
              )
          )
      ),
  )
  try:
    response = gemini_client.models.generate_content(
        model=models.Models.GEMINI_2_5_FLASH_TTS.value,
        contents=types.Content(parts=[types.Part(text=text)], role='user'),
        config=config,
    )

    if (
        not response.candidates
        or not response.candidates[0].content
        or not response.candidates[0].content.parts
    ):
      llm_logging.log_operation_end('Generate Audio (TTS)', success=False)
      raise opal_adk_error.OpalAdkError(
          status_message='No audio generated.',
          status_code=code_pb2.INTERNAL,
          details='generate_audio: TTS model returned no audio data.',
      )

    pcm_data = response.candidates[0].content.parts[0].inline_data.data  # pytype: disable=attribute-error

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
      wf.setnchannels(1)
      wf.setsampwidth(2)
      wf.setframerate(24000)
      wf.writeframes(pcm_data)

    llm_logging.log_operation_end('Generate Audio (TTS)', success=True)
    return wav_buffer.getvalue(), 'audio/wav'

  except Exception as e:
    if isinstance(e, opal_adk_error.OpalAdkError):
      raise
    logging.exception('Failed to generate audio: %s', e)
    llm_logging.log_operation_end('Generate Audio (TTS)', success=False)
    raise opal_adk_error.OpalAdkError(
        status_message='Failed to generate audio.',
        status_code=code_pb2.INTERNAL,
        internal_details=str(e),
        details='generate_audio: An error occurred during audio generation.',
    )
