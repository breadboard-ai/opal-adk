"""Model constraints enum."""

import enum


class ModelConstraint(enum.Enum):
  """The model constraint to use for generation."""
  UNSPECIFIED = "unspecified"
  TEXT_FLASH = "text_flash"
  TEXT_PRO = "text_pro"
  IMAGE = "image"
  VIDEO = "video"
  SPEECH = "speech"
  MUSIC = "music"


def model_constraint_from_string(val: str) -> ModelConstraint:
  """Converts a string to a ModelConstraint."""
  try:
    return ModelConstraint[val.upper()]
  except KeyError:
    return ModelConstraint.UNSPECIFIED


def model_constraint_to_string(val: ModelConstraint) -> str:
  """Converts a ModelConstraint to a string."""
  return val.value
