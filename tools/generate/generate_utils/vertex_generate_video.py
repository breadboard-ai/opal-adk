
"""Utilities for interacting with the Gemini API."""

import logging
import time

from google.api_core import exceptions as api_core_exceptions
from google.genai import errors as genai_errors
from google.genai import types
from opal_adk.clients import vertex_ai_client
from opal_adk.error_handling import opal_adk_error
from opal_adk.types import image_types
from opal_adk.types import models

from google.rpc import code_pb2


Part = types.Part

RETRIABLE_RESPONSE_ERRORS = (
    api_core_exceptions.TooManyRequests,
    api_core_exceptions.ResourceExhausted,
    api_core_exceptions.RetryError,
    genai_errors.ServerError,
)


def _generate_video_impl(
    text_prompt: str,
    reference_image_parts: list[types.Part] | None,
    disable_prompt_rewrite: bool,
    aspect_ratio: image_types.AspectRatio,
    duration_seconds: int,
    model_name: str,
) -> tuple[bytes, str]:
  """Internal implementation of video generation."""

  def _get_video_image_config(
      current_model: str,
  ) -> tuple[
      types.Image | None,
      list[types.VideoGenerationReferenceImage] | None,
  ]:
    """Returns (base_image, reference_images) for the given model."""
    if not reference_image_parts:
      return None, None
    if current_model in (
        models.Models.VEO_3_1.value,
        models.Models.VEO_3_1_FAST.value,
    ):
      ref_images = [
          types.VideoGenerationReferenceImage(
              image=types.Image(
                  image_bytes=part.inline_data.data,
                  mime_type=part.inline_data.mime_type,
              ),
              reference_type='ASSET',
          )
          for part in reference_image_parts
          if part.inline_data
      ]
      return None, ref_images
    else:
      first_part = reference_image_parts[0]
      if first_part.inline_data:
        return (
            types.Image(
                image_bytes=first_part.inline_data.data,
                mime_type=first_part.inline_data.mime_type,
            ),
            None,
        )
      return None, None

  fall_backs = [
      model_name,
      models.Models.VEO_3.value,
      models.Models.VEO_3_FAST.value,
  ]
  # Deduplicate fallbacks while preserving order
  fall_backs = list(dict.fromkeys(fall_backs))

  operation = None
  client = vertex_ai_client.create_vertex_ai_client()

  for current_model in fall_backs:
    current_base_image, current_reference_images = _get_video_image_config(
        current_model
    )

    config = types.GenerateVideosConfig(
        aspect_ratio=aspect_ratio.value,
        person_generation='allow_adult',
        enhance_prompt='no' if disable_prompt_rewrite else 'yes',
        duration_seconds=duration_seconds,
        reference_images=current_reference_images,
    )

    try:
      operation = client.models.generate_videos(
          model=current_model,
          prompt=text_prompt,
          image=current_base_image,
          config=config,
      )
      break
    except RETRIABLE_RESPONSE_ERRORS as e:
      logging.warning(
          'Retriable error when calling Veo with model %s: %s, '
          'retrying on inferior model.',
          current_model,
          e,
      )
      continue
    except Exception as e:  # pylint: disable=broad-exception-caught
      if 'RESOURCE_EXHAUSTED' in str(e):
        logging.warning(
            'RESOURCE_EXHAUSTED when generating video with model %s: %s, '
            'trying next model.',
            current_model,
            e,
        )
        continue
      logging.exception(
          'Failed to create video LRO with model %s: %s',
          current_model,
          e,
      )
      raise opal_adk_error.OpalAdkError(
          status_message='Failed to initiate video request.',
          status_code=code_pb2.INTERNAL,
          details='Internal error occurred.',
      ) from e

  if not operation:
    raise opal_adk_error.OpalAdkError(
        status_message='Failed to initiate video request.',
        status_code=code_pb2.INTERNAL,
        details='Internal error occurred.',
    )

  while not operation.done:
    time.sleep(3)
    operation = client.operations.get(operation)

  # Check response
  if operation.error:
    logging.warning('Vertex Veo failed to generate video: %s', operation.error)
    raise opal_adk_error.OpalAdkError(
        status_message=(
            'No videos generated:'
            f' {operation.error.get("message", "Unknown error")}.'
        ),
        status_code=operation.error.get('code', code_pb2.INTERNAL),
        details=operation.error.get('message', 'Unknown error'),
    )

  res = None
  if operation.response and operation.response.generated_videos:
    res = operation.response.generated_videos[0].video
    if not res:
      # Check rai_media_filtered_reasons
      if operation.response.rai_media_filtered_reasons:
        raise opal_adk_error.OpalAdkError(
            status_message='No videos generated.',
            status_code=code_pb2.INVALID_ARGUMENT,
            details=operation.response.rai_media_filtered_reasons[0],
        )

  if not res:
    raise opal_adk_error.OpalAdkError(
        status_message='No videos generated by Veo Vertex backend.',
        status_code=code_pb2.INTERNAL,
    )
  else:
    return res.video_bytes, res.mime_type


def generate_video_via_vertex_ai(
    text_prompt: str,
    reference_image_parts: list[types.Part] | None = None,
    disable_prompt_rewrite: bool = False,
    aspect_ratio: image_types.AspectRatio = image_types.AspectRatio.RATIO_16_9,
    duration_seconds: int = 8,
    model_name: str | None = None,
) -> tuple[bytes, str]:
  """Generates video with Veo on Cloud Vertex AI.

  Args:
    text_prompt: The text prompt for video generation.
    reference_image_parts: Optional list of genai.Parts containing reference
      images. For veo-3.1 models, up to 3 images are used as reference images.
      For other models, only the first image is used as base image.
    disable_prompt_rewrite: Whether to disable automatic prompt rewriting.
    aspect_ratio: Video aspect ratio ('16:9' or '9:16').
    duration_seconds: Duration of the generated video.
    model_name: The VEO model to use. Defaults to MODEL_VEO_3_FAST.

  Returns:
    A tuple of (video_bytes, mime_type).
  """
  logging.info('Starting Generate Video (Veo)')
  if aspect_ratio not in (
      image_types.AspectRatio.RATIO_16_9,
      image_types.AspectRatio.RATIO_9_16,
  ):
    aspect_ratio = image_types.AspectRatio.RATIO_16_9

  if model_name is None:
    model_name = models.Models.VEO_3_FAST.value

  try:
    video_bytes, mime_type = _generate_video_impl(
        text_prompt=text_prompt,
        reference_image_parts=reference_image_parts,
        disable_prompt_rewrite=disable_prompt_rewrite,
        aspect_ratio=aspect_ratio,
        duration_seconds=duration_seconds,
        model_name=model_name,
    )
    logging.info('Generate Video (Veo) completed successfully.')
    return video_bytes, mime_type
  except Exception as e:
    logging.error('Video generation failed: %s', e)
    raise
