"""Defines an Enum containing identifiers for various models used within OPAL ADK."""

import enum


class Models(enum.Enum):
  """Supported models."""

  MODEL_GEMINI_2_0_FLASH = 'gemini-2.0-flash'
  MODEL_GEMINI_2_0_FLASH_LITE = 'gemini-2.0-flash-lite'
  MODEL_GEMINI_2_5_PRO = 'gemini-2.5-pro'
  MODEL_GEMINI_2_5_FLASH = 'gemini-2.5-flash'
  MODEL_GEMINI_2_5_FLASH_LITE = 'gemini-2.5-flash-lite'
  MODEL_IMAGEN = 'imagen-4.0-generate-001'
  MODEL_IMAGEN_3 = 'imagen-3.0-generate-002'
  MODEL_IMAGEN_3_FAST = 'imagen-3.0-fast-generate-001'
  MODEL_GEMINI_IMAGE = 'gemini-2.0-flash-preview-image-generation'
  MODEL_GEMINI_2_5_IMAGE = 'gemini-2.5-flash-image-preview'
  MODEL_VEO_2 = 'veo-2.0-generate-001'
  MODEL_VEO_3_FAST = 'veo-3.0-fast-generate-preview'
  MODEL_VEO_3 = 'veo-3.0-generate-preview'
  MODEL_GEMINI_2_5_FLASH_TTS = 'gemini-2.5-flash-preview-tts'
