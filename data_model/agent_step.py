"""Data model for an Opal plan.

An OpalPlan is a collection of steps that define a sequence of operations
to be performed.
"""

import dataclasses

from google.genai import types
from opal_adk.types import model_constraint as mc
from opal_adk.types import ui_type as ut


@dataclasses.dataclass(frozen=True)
class AgentStep:
  """Represents a single step in an e2e Opal plan when in 'agent mode'."""

  step_name: str
  objective: types.Content
  ui_prompt: types.Content
  model_constraint: mc.ModelConstraint = (
      mc.ModelConstraint.UNSPECIFIED
  )
  invocation_id: str | None = None
  input_parameters: list[str] = dataclasses.field(default_factory=list)
  output: str = ''
  reasoning: str = ''
  ui_type: ut.UIType = ut.UIType.UNSPECIFIED
  is_list_output: bool = False
  system_prompt: str = ''

  def render(self, include_system_prompt: bool = False) -> str:
    """Renders the PlanStep as a string."""
    system_prompt_step = (
        f'<system_prompt>{self.system_prompt}</system_prompt>'
        if include_system_prompt and self.system_prompt
        else '<system_prompt></system_prompt>'
    )
    return f"""<plan_step>
      <step_name>{self.step_name}</step_name>
      <objective>{self.objective}</objective>
      {system_prompt_step}
      <model_constraint>{self.model_constraint.value}</model_constraint>
      <invocation_id>{self.invocation_id}</invocation_id>
      <input_parameters>{self.input_parameters}</input_parameters>
      <output>{self.output}</output>
      <reasoning>{self.reasoning}</reasoning>
      <ui_type>{self.ui_type.value}</ui_type>
      <ui_prompt>{self.ui_prompt}</ui_prompt>
      <is_list_output>{self.is_list_output}</is_list_output>
      </plan_step>"""

  def render_as_input_parameter(self) -> str:
    """Renders the PlanStep as an input parameter."""
    return f"""<application_input>
      <field_name>{self.step_name}</field_name>
      <field_type>{self.ui_type}</field_type>
      <field_blurb>{self.objective}</field_blurb>
      </application_input>"""
