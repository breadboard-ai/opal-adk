"""Data classes for configuring safety settings.

This file defines data classes for specifying safety settings for text and image
content within an application, including harm block thresholds for various
categories of text content and a general safety level for images.
"""

import dataclasses
from typing import Literal


# TODO(b/462420103) - Switch to use enums here and elsewhere.
@dataclasses.dataclass
class TextSafetySettings:
  """Safety settings for text generation."""

  _harm_block_threshold = (
      Literal[
          'OFF',
          'BLOCK_NONE',
          'BLOCK_LOW_AND_ABOVE',
          'BLOCK_MEDIUM_AND_ABOVE',
          'BLOCK_ONLY_HIGH',
      ]
      | None
  )

  harassment_threshold: _harm_block_threshold = None
  hate_speech_threshold: _harm_block_threshold = None
  sexually_explicit_threshold: _harm_block_threshold = None
  dangerous_content_threshold: _harm_block_threshold = None


@dataclasses.dataclass
class SafetySettings:
  """Safety settings for the application."""

  image_safety_level: (
      Literal['block_most', 'block_some', 'block_few', 'block_fewest'] | None
  ) = None
  text_safety_settings: TextSafetySettings | None = None
