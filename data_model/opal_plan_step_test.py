"""Unit tests for opal_plan_step."""

import textwrap
from opal_adk.data_model import opal_plan_step
from google3.testing.pybase import googletest


class OpalPlanStepTest(googletest.TestCase):

  def test_render_without_system_prompt(self):
    step = opal_plan_step.OpalPlanStep(
        step_name="step1",
        step_intent="intent1",
        model_api="api1",
        input_parameters=["param1"],
        output="output1",
        reasoning="reasoning1",
        system_prompt="system_prompt1",
    )
    expected = textwrap.dedent("""
      <plan_step>
      <step_name>step1</step_name>
      <step_intent>intent1</step_intent>
      <system_prompt></system_prompt>
      <model_api>api1</model_api>
      <input_parameters>['param1']</input_parameters>
      <output>output1</output>
      <reasoning>reasoning1</reasoning>
      <is_list_output>False</is_list_output>
      </plan_step>
      """)
    self.assertEqual(expected, step.render(include_system_prompt=False))

  def test_render_with_system_prompt_but_empty(self):
    step = opal_plan_step.OpalPlanStep(
        step_name="step1",
        step_intent="intent1",
        model_api="api1",
        input_parameters=["param1"],
        output="output1",
        reasoning="reasoning1",
        system_prompt="",
    )
    expected = textwrap.dedent("""
      <plan_step>
      <step_name>step1</step_name>
      <step_intent>intent1</step_intent>
      <system_prompt></system_prompt>
      <model_api>api1</model_api>
      <input_parameters>['param1']</input_parameters>
      <output>output1</output>
      <reasoning>reasoning1</reasoning>
      <is_list_output>False</is_list_output>
      </plan_step>
      """)
    self.assertEqual(expected, step.render(include_system_prompt=True))

  def test_render_with_system_prompt(self):
    step = opal_plan_step.OpalPlanStep(
        step_name="step1",
        step_intent="intent1",
        model_api="api1",
        input_parameters=["param1"],
        output="output1",
        reasoning="reasoning1",
        system_prompt="system_prompt1",
    )
    expected = textwrap.dedent("""
      <plan_step>
      <step_name>step1</step_name>
      <step_intent>intent1</step_intent>
      <system_prompt>system_prompt1</system_prompt>
      <model_api>api1</model_api>
      <input_parameters>['param1']</input_parameters>
      <output>output1</output>
      <reasoning>reasoning1</reasoning>
      <is_list_output>False</is_list_output>
      </plan_step>
      """)
    self.assertEqual(expected, step.render(include_system_prompt=True))

  def test_render_as_input_parameter_image_generation(self):
    step = opal_plan_step.OpalPlanStep(
        step_name="step1",
        step_intent="intent1",
        model_api="image_generation",
        input_parameters=["param1"],
        output="output1",
        reasoning="reasoning1",
    )
    expected = textwrap.dedent("""
        <application_input>
        <field_name>step1</field_name>
        <field_type>image</field_type>
        <field_blurb>intent1</field_blurb>
        </application_input>
    """)
    self.assertEqual(expected, step.render_as_input_parameter())

  def test_render_as_input_parameter_ai_image_tool(self):
    step = opal_plan_step.OpalPlanStep(
        step_name="step1",
        step_intent="intent1",
        model_api="ai_image_tool",
        input_parameters=["param1"],
        output="output1",
        reasoning="reasoning1",
    )
    expected = textwrap.dedent("""
        <application_input>
        <field_name>step1</field_name>
        <field_type>image</field_type>
        <field_blurb>intent1</field_blurb>
        </application_input>
    """)
    self.assertEqual(expected, step.render_as_input_parameter())

  def test_render_as_input_parameter_tts(self):
    step = opal_plan_step.OpalPlanStep(
        step_name="step1",
        step_intent="intent1",
        model_api="tts",
        input_parameters=["param1"],
        output="output1",
        reasoning="reasoning1",
    )
    expected = textwrap.dedent("""
        <application_input>
        <field_name>step1</field_name>
        <field_type>audio</field_type>
        <field_blurb>intent1</field_blurb>
        </application_input>
    """)
    self.assertEqual(expected, step.render_as_input_parameter())

  def test_render_as_input_parameter_text(self):
    step = opal_plan_step.OpalPlanStep(
        step_name="step1",
        step_intent="intent1",
        model_api="other",
        input_parameters=["param1"],
        output="output1",
        reasoning="reasoning1",
    )
    expected = textwrap.dedent("""
        <application_input>
        <field_name>step1</field_name>
        <field_type>text</field_type>
        <field_blurb>intent1</field_blurb>
        </application_input>
    """)
    self.assertEqual(expected, step.render_as_input_parameter())


if __name__ == "__main__":
  googletest.main()
