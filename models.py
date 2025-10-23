"""Defines Enum identifiers for supported models."""

import enum


class Models(enum.Enum):
  """Enum identifiers for supported models."""
  GEMINI_2_0_FLASH = "gemini-2.0-flash"
  GEMINI_2_5_FLASH = "gemini-2.5-flash"
  GEMINI_2_5_PRO = "gemini-2.5-pro"
