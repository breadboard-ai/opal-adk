"""Utilities for interacting with the Gemini API."""

import io
import logging
import wave

from google.api_core import exceptions as api_core_exceptions
from google.genai import errors as genai_errors
from google.genai import types
from opal_adk.clients import vertex_ai_client
from opal_adk.error_handling import opal_adk_error
from opal_adk.types import image_types
from opal_adk.types import models
from opal_adk.util import llm_logging

from google.rpc import code_pb2


Part = types.Part

RETRIABLE_RESPONSE_ERRORS = (
    api_core_exceptions.TooManyRequests,
    api_core_exceptions.ResourceExhausted,
    api_core_exceptions.RetryError,
    genai_errors.ServerError,
)


def extract_grounding_metadata(
    response: types.GenerateContentResponse,
) -> list[types.Content]:
  """Extracts grounding metadata from Vertex AI generate_content response."""
  result = []
  candidate = response.candidates[0]
  if not candidate.grounding_metadata:
    return result
  logging.info('Grounding metadata: %s', candidate.grounding_metadata)
  if candidate.grounding_metadata.web_search_queries:
    result.append(
        types.Content(
            parts=[
                Part(
                    text='\n\n\nRelated Google Search queries: '
                    + ', '.join(
                        candidate.grounding_metadata.web_search_queries
                    ),
                )
            ]
        )
    )
  if candidate.grounding_metadata.grounding_chunks:
    grounding_chunks = '\n\nSources:\n'
    for grounding_chunk in candidate.grounding_metadata.grounding_chunks:
      if not grounding_chunk.web:
        logging.info('Unexpected grounding chunk: %s', grounding_chunk)
      else:
        grounding_chunks += (
            f'{grounding_chunk.web.title}: {grounding_chunk.web.uri}\n'
        )
    result.append(
        types.Content(
            parts=[Part(text=grounding_chunks)],
        )
    )
  return result


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
        contents=types.Content(parts=[types.Part(text=text, role='user')]),
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
          details='TTS model returned no audio data.',
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
        details='An error occurred during audio generation.',
    )


def _image_bytes_from_generated_image(
    generated_image: types.GeneratedImage,
) -> tuple[bytes, str]:
  """Returns the image bytes and mime type from a GeneratedImage.

  Args:
    generated_image: The generated image object from Gemini API.

  Returns:
    Tuple of (image_bytes, mime_type).

  Raises:
    opal_adk_error.OpalAdkError: If no image was returned.
  """
  if generated_image.image and generated_image.image.image_bytes:
    mime_type = generated_image.image.mime_type or 'image/png'
    return generated_image.image.image_bytes, mime_type
  raise opal_adk_error.OpalAdkError(
      status_message='No images generated.',
      status_code=code_pb2.INTERNAL,
      details='This may indicate an invalid or policy violating prompt.',
  )


def generate_image_via_genai_api(
    image_prompt: str,
    aspect_ratio: image_types.AspectRatio = image_types.AspectRatio.RATIO_1_1,
    image_safety_level: image_types.ImageSafetyLevel | None = None,
) -> tuple[bytes, str]:
  """Generates an image with Gemini API.

  Args:
    image_prompt: The text prompt describing the desired image.
    aspect_ratio: The aspect ratio of the generated image.
    image_safety_level: Safety filtering level for the generated image.

  Returns:
    Tuple of (image_bytes, mime_type).

  Raises:
    opal_adk_error.OpalAdkError: If unable to generate image.
  """
  logging.info('Starting Generate Image (Imagen)')
  last_exception = None

  # The models we will attempt generation with
  model_imagen_3_fallbacks = [
      models.Models.IMAGEN_3.value,
      models.Models.IMAGEN_3_FAST.value,
  ]

  client = vertex_ai_client.create_vertex_ai_client()

  for model_name in model_imagen_3_fallbacks:
    try:
      logging.info('Generating image with %s', model_name)
      images = client.models.generate_images(
          model=model_name,
          prompt=image_prompt,
          config=types.GenerateImagesConfig(
              number_of_images=1,
              language='en',
              aspect_ratio=aspect_ratio.value,
              safety_filter_level=image_safety_level.value
              if image_safety_level
              else None,
              person_generation='allow_all',
          ),
      )
      if images:
        logging.info('Generate Image (Imagen) completed successfully.')
        return _image_bytes_from_generated_image(images[0])
    except RETRIABLE_RESPONSE_ERRORS as e:
      last_exception = e
      logging.warning(
          'Failed to generate image with model %s: %s. Trying next model.',
          model_name,
          e,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      if 'RESOURCE_EXHAUSTED' in str(e):
        logging.warning(
            'Failed to generate image with model %s: %s. Trying next model.',
            model_name,
            e,
        )
        last_exception = e
        continue
      logging.error('Generate Image (Imagen) failed: %s', e)
      raise opal_adk_error.get_opal_adk_error(e) from e

  logging.error('Generate Image (Imagen) failed to generate any images.')
  if last_exception:
    raise opal_adk_error.get_opal_adk_error(last_exception)

  raise opal_adk_error.OpalAdkError(
      status_message='No images generated.',
      status_code=code_pb2.INTERNAL,
      details='This may indicate an invalid or policy violating prompt.',
  )
