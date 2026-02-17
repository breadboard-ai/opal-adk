"""Utility function to edit/generate image via Gemini API."""

import logging

from google.genai import types
from opal_adk.clients import vertex_ai_client
from opal_adk.error_handling import opal_adk_error
from opal_adk.types import image_types
from opal_adk.types import models
from opal_adk.util import gemini_utils
from opal_adk.util import llm_logging

from google.rpc import code_pb2


def gemini_generate_images(
    parts: list[types.Part],
    aspect_ratio: image_types.AspectRatio = image_types.AspectRatio.RATIO_1_1,
    model_name: str | None = None,
    safety_settings: list[types.SafetySetting] | None = None,
) -> list[tuple[bytes, str]]:
  """Edits/Generates an image with Gemini API.

  Args:
    parts: List of genai Parts (text and/or inline_data) to send to Gemini.
    aspect_ratio: The aspect ratio of the generated image.
    model_name: The Gemini model to use. Defaults to GEMINI_2_5_IMAGE.
    safety_settings: Optional safety settings.

  Returns:
    List of tuples (image_bytes, mime_type) for each generated image.

  Raises:
    opal_adk_error.OpalAdkError: If unable to generate image.
  """
  llm_logging.log_operation_start('Generate Image (Gemini)')
  logging.info('Gemini image out request parts count: %d', len(parts))

  if model_name is None:
    model_name = models.Models.GEMINI_2_5_FLASH_IMAGE.value

  config = types.GenerateContentConfig(
      response_modalities=['TEXT', 'IMAGE'],
      safety_settings=safety_settings,
      image_config=types.ImageConfig(
          aspect_ratio=aspect_ratio.value,
      ),
  )

  client = vertex_ai_client.create_vertex_ai_client(use_vertex=False)

  try:
    logging.info(
        'gemini_generate_image: Generating image with config %s',
        config,
    )
    response = client.models.generate_content(
        model=model_name,
        contents=types.Content(parts=parts, role='user'),
        config=config,
    )
  except Exception as e:
    logging.warning('gemini_generate_image: Unable to generate image: %s.', e)
    llm_logging.log_operation_end('Generate Image (Gemini)', success=False)
    raise opal_adk_error.get_opal_adk_error(e) from e

  if not response.candidates:
    llm_logging.log_operation_end('Generate Image (Gemini)', success=False)
    raise opal_adk_error.OpalAdkError(
        status_message='No candidates returned from Gemini API.',
        status_code=code_pb2.INTERNAL,
    )

  content = response.candidates[0].content
  if not content or not content.parts:
    llm_logging.log_operation_end('Generate Image (Gemini)', success=False)
    raise opal_adk_error.OpalAdkError(
        status_message='No content parts returned from Gemini API.',
        status_code=code_pb2.INTERNAL,
    )

  gemini_utils.validate_candidate_recitation(response)
  image_parts = []
  text_parts = []
  for part in content.parts:
    if part.inline_data is not None:
      image_parts.append((part.inline_data.data, part.inline_data.mime_type))
    if part.text:
      text_parts.append(part.text)

  if not image_parts:
    message = 'No images generated.'
    details = ''
    if text_parts:
      details = 'Gemini returned the following text instead of image(s): '
      details += ' '.join(text_parts)
    llm_logging.log_operation_end('Generate Image (Gemini)', success=False)
    raise opal_adk_error.OpalAdkError(
        status_message=message,
        status_code=code_pb2.INTERNAL,
        details=details,
        internal_details='gemini_generate_image: ' + details,
    )

  llm_logging.log_operation_end('Generate Image (Gemini)', success=True)
  return image_parts
