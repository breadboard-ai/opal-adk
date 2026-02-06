"""Valid input types for Opal ADK."""

import enum


class ValidInputTypes(enum.Enum):
  ANY = "any"
  TEST = "text"
  FILE_UPLOAD = "file-upload"
