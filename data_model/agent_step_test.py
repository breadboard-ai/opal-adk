"""Tests for agent_step."""

from absl.testing import absltest
from opal_adk.data_model import agent_step


class AgentStepTest(absltest.TestCase):

  def test_render(self):
    step = agent_step.AgentStep(
        step_name='step1',
        objective='objective1',
        model_constraint='constraint1',
        invocation_id='inv1',
        input_parameters=['param1'],
        output='output1',
        reasoning='reasoning1',
        ui_type='ui_type1',
        ui_prompt='ui_prompt1',
        is_list_output=True,
        system_prompt='sys_prompt1',
    )
    expected_output = """<plan_step>
      <step_name>step1</step_name>
      <objective>objective1</objective>
      <system_prompt>sys_prompt1</system_prompt>
      <model_constraint>constraint1</model_constraint>
      <invocation_id>inv1</invocation_id>
      <input_parameters>['param1']</input_parameters>
      <output>output1</output>
      <reasoning>reasoning1</reasoning>
      <ui_type>ui_type1</ui_type>
      <ui_prompt>ui_prompt1</ui_prompt>
      <is_list_output>True</is_list_output>
      </plan_step>"""

    self.assertEqual(
        step.render(include_system_prompt=True).strip(), expected_output.strip()
    )

  def test_render_no_system_prompt(self):
    step = agent_step.AgentStep(
        step_name='step1',
        objective='objective1',
        model_constraint='constraint1',
        ui_prompt='prompt',
    )
    # Check that system prompt is empty tag by default
    self.assertIn('<system_prompt></system_prompt>', step.render())

  def test_render_as_input_parameter(self):
    step = agent_step.AgentStep(
        step_name='step1',
        objective='objective1',
        ui_type='ui_type1',
        ui_prompt='prompt',
    )
    expected_output = """<application_input>
      <field_name>step1</field_name>
      <field_type>ui_type1</field_type>
      <field_blurb>objective1</field_blurb>
      </application_input>"""
    self.assertEqual(
        step.render_as_input_parameter().strip(), expected_output.strip()
    )


if __name__ == '__main__':
  absltest.main()
