"""Defines Enum identifiers for supported models."""

import enum


class SimpleModel(enum.Enum):
  """The simple model types used within node_agent functions."""

  PRO = "pro"
  FLASH = "flash"
  LITE = "light"


class Models(enum.Enum):
  """Enum identifiers for supported models."""
  PRO_MODEL_NAME = "gemini-3-pro-preview"
  LITE_MODEL_NAME = "gemini-2.5-flash-lite"
  FLASH_MODEL_NAME = "gemini-2.5-flash"
  GEMINI_2_0_FLASH = "gemini-2.0-flash"
  GEMINI_2_5_FLASH = "gemini-2.5-flash"
  GEMINI_2_5_FLASH_TTS = "gemini-2.5-flash-preview-tts"
  GEMINI_2_5_PRO = "gemini-2.5-pro"
  GEMINI_3_FLASH = "gemini-3-flash-preview"
  GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite"
  GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
  IMAGEN = "imagen-4.0-generate-001"
  IMAGEN_3 = "imagen-3.0-generate-002"
  IMAGEN_3_FAST = "imagen-3.0-fast-generate-001"
  GEMINI_IMAGE = "gemini-2.0-flash-preview-image-generation"
  GEMINI_2_5_IMAGE = "gemini-2.5-flash-image-preview"
  VEO_2 = "veo-2.0-generate-001"
  VEO_3_FAST = "veo-3.0-fast-generate-preview"
  VEO_3 = "veo-3.0-generate-preview"
  VEO_3_1 = "veo-3.1-generate-preview"
  VEO_3_1_FAST = "veo-3.1-fast-generate-preview"


def simple_model_to_model(simple_model: SimpleModel) -> Models:
  """Converts a SimpleModel enum to its corresponding Models enum.

  Args:
    simple_model: The SimpleModel to convert.

  Returns:
    The corresponding Models enum value.
  """
  match simple_model:
    case SimpleModel.PRO:
      return Models.PRO_MODEL_NAME
    case SimpleModel.LITE:
      return Models.LITE_MODEL_NAME
    case SimpleModel.FLASH:
      return Models.FLASH_MODEL_NAME
