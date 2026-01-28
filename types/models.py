"""Defines Enum identifiers for supported models."""

import enum


class SimpleModel(enum.Enum):
  """The simple model types used within node_agent functions."""

  PRO = "pro"
  FLASH = "flash"
  LITE = "light"


class Models(enum.Enum):
  """Enum identifiers for supported models."""
  GEMINI_2_0_FLASH = "gemini-2.0-flash"
  GEMINI_2_5_FLASH = "gemini-2.5-flash"
  GEMINI_2_5_PRO = "gemini-2.5-pro"
  PRO_MODEL_NAME = "gemini-3-pro-preview"
  LITE_MODEL_NAME = "gemini-2.5-flash-lite"
  FLASH_MODEL_NAME = "gemini-2.5-flash"


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
