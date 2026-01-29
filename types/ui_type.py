"""UI types enum."""

import enum


class UIType(enum.Enum):
  """The UI type of the agent mode node."""
  UNSPECIFIED = "unspecified"
  CHAT = "chat"
  A2UI = "a2ui"
  NONE = "none"


def ui_type_from_string(val: str) -> UIType:
  """Converts a string to a UIType."""
  try:
    return UIType[val.upper()]
  except KeyError:
    return UIType.UNSPECIFIED


def ui_type_to_string(val: UIType) -> str:
  """Converts a UIType to a string."""
  return val.value
