"""Generates speech audio from text using Google's GenAI models.

This module provides a function to convert a sequence of text strings into
audio content, utilizing specified voice types. It handles voice mapping
and error management.
"""

import enum
import logging
from google.genai import types
from opal_adk.error_handling import opal_adk_error
from opal_adk.tools.generate.generate_utils import vertex_generate_audio
from google.rpc import code_pb2


class VoiceTypes(enum.Enum):
  EN_US_MALE = 'en_US_male'
  EN_US_FEMALE = 'en_US_female'


class VoiceNames(enum.Enum):
  CHARON = 'Charon'
  ZEPHYR = 'Zephyr'


_VOICE_MAPPING = {
    VoiceTypes.EN_US_MALE: VoiceNames.CHARON,
    VoiceTypes.EN_US_FEMALE: VoiceNames.ZEPHYR,
}


def generate_speech_from_text(
    tts_input: list[str], voice: str = VoiceTypes.EN_US_FEMALE.value
) -> list[types.Content]:
  """Generates speech from text.

  Args:
   tts_input: A sequence of texts to turn into speech.
   voice: The voice to use for speech generation. This must be either
     'en_US_male' or 'en_US_female'.

  Returns:
    A list of `types.Content`, where each element contains the generated audio
    for the corresponding input text in `tts_input`.
  """

  try:
    voice_type = VoiceTypes(voice)
  except ValueError:
    error_text = (
        f'Received invalid voice option, received {voice} but must be one of'
        f' {[v.value for v in VoiceTypes]}.'
    )
    raise opal_adk_error.OpalAdkError(
        logged=f'generated_speech_from_text: {error_text}',
        status_message=error_text,
        status_code=code_pb2.INVALID_ARGUMENT,
    ) from None

  voice_name = _VOICE_MAPPING[voice_type].value
  logging.info('Calling TTS voice %s', voice_name)
  try:
    results = []
    for text in tts_input:
      audio_bytes, mime_type = (
          vertex_generate_audio.generate_audio(text, voice_name)
      )
      results.append(
          types.Content(
              parts=[
                  types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
              ]
          )
      )
    return results
  except Exception as e:  # pylint: disable=broad-exception-caught
    raise opal_adk_error.OpalAdkError(
        logged=f'Error generating audio via TTS: error type: {type(e)}: {e}',
        status_message=opal_adk_error.MODEL_CALL_ERROR_MESSAGE,
        status_code=code_pb2.INTERNAL,
        internal_details=f'Error generating audio via TTS: {e}',
    )
