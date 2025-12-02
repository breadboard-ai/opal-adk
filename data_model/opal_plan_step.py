"""Data models for representing steps within an Opal plan."""

import dataclasses
import textwrap
from opal_adk.data_model import step_execution_options

_StepExecutionOptions = step_execution_options.StepExecutionOptions


@dataclasses.dataclass(frozen=True)
class OpalPlanStep:
  """Represents a single step in an e2e Opal plan."""

  step_name: str
  step_intent: str
  model_api: str
  input_parameters: list[str]
  output: str
  reasoning: str
  is_list_output: bool = False
  options: _StepExecutionOptions = dataclasses.field(
      default_factory=_StepExecutionOptions
  )
  system_prompt: str = ''

  def render(self, include_system_prompt: bool = False) -> str:
    """Renders the PlanStep as a string."""
    system_prompt_step = (
        f'<system_prompt>{self.system_prompt}</system_prompt>'
        if include_system_prompt and self.system_prompt
        else '<system_prompt></system_prompt>'
    )
    return textwrap.dedent(f"""
      <plan_step>
      <step_name>{self.step_name}</step_name>
      <step_intent>{self.step_intent}</step_intent>
      {system_prompt_step}
      <model_api>{self.model_api}</model_api>
      <input_parameters>{self.input_parameters}</input_parameters>
      <output>{self.output}</output>
      <reasoning>{self.reasoning}</reasoning>
      <is_list_output>{self.is_list_output}</is_list_output>
      </plan_step>
      """)

  def render_as_input_parameter(self) -> str:
    """Renders the PlanStep as an input parameter."""
    if self.model_api in ('image_generation', 'ai_image_tool'):
      output_type = 'image'
    elif self.model_api == 'tts':
      output_type = 'audio'
    else:
      output_type = 'text'
    return textwrap.dedent(f"""
        <application_input>
        <field_name>{self.step_name}</field_name>
        <field_type>{output_type}</field_type>
        <field_blurb>{self.step_intent}</field_blurb>
        </application_input>
    """)
