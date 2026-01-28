"""Defines the StepExecutionOptions dataclass for advanced execution settings."""

import dataclasses
from typing import Literal
from opal_adk.data_model import safety_settings
from opal_adk.types import models

_SafetySettings = safety_settings.SafetySettings


@dataclasses.dataclass
class StepExecutionOptions:
  """Advanced options for an execution step."""

  chat_with_user: bool = False
  plan_mode: Literal['All at once', 'Go in order', 'Think as I go'] | None = (
      None
  )
  render_mode: Literal['Manual', 'Markdown', 'HTML', 'Interactive'] | None = (
      None
  )

  model_name: str = models.Models.GEMINI_2_5_FLASH.value
  auto_render: bool = True
  disable_prompt_rewrite: bool = False
  retrieval_mode: str = ''
  system_instruction: str = ''
  safety_settings: _SafetySettings = dataclasses.field(
      default_factory=_SafetySettings
  )
