import logging

from google.api_core import exceptions as api_core_exceptions
from google.genai import errors as genai_errors
from google.genai import types
from opal_adk.clients import vertex_ai_client
from opal_adk.error_handling import opal_adk_error
from opal_adk.types import image_types
from opal_adk.types import models

from google.rpc import code_pb2


RETRIABLE_RESPONSE_ERRORS = (
    api_core_exceptions.TooManyRequests,
    api_core_exceptions.ResourceExhausted,
    api_core_exceptions.RetryError,
    genai_errors.ServerError,
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
