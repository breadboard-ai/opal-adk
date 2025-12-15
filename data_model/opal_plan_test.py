"""Tests for opal_plan."""

from absl.testing import absltest
from absl.testing import parameterized
from opal_adk.data_model import opal_plan
from opal_adk.data_model import opal_plan_step


class OpalPlanTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'single_step',
          [
              opal_plan_step.OpalPlanStep(
                  step_name='step1',
                  step_intent='intent1',
                  model_api='api1',
                  input_parameters=['param1'],
                  output='output1',
                  reasoning='reasoning1',
              )
          ],
          [
              '<plan>',
              '  <plan_name>test_plan</plan_name>',
              '  <plan_steps>',
              '    <plan_step>',
              '      <step_name>step1</step_name>',
              '      <step_intent>intent1</step_intent>',
              '      <system_prompt></system_prompt>',
              '      <model_api>api1</model_api>',
              "      <input_parameters>['param1']</input_parameters>",
              '      <output>output1</output>',
              '      <reasoning>reasoning1</reasoning>',
              '      <iterations>1</iterations>',
              '      <is_list_output>False</is_list_output>',
              '    </plan_step>',
              '  </plan_steps>',
              '</plan>',
          ],
      ),
      (
          'parallel_steps',
          [[
              opal_plan_step.OpalPlanStep(
                  step_name='step1',
                  step_intent='intent1',
                  model_api='api1',
                  input_parameters=['param1'],
                  output='output1',
                  reasoning='reasoning1',
              ),
              opal_plan_step.OpalPlanStep(
                  step_name='step2',
                  step_intent='intent2',
                  model_api='api2',
                  input_parameters=['param2'],
                  output='output2',
                  reasoning='reasoning2',
              ),
          ]],
          [
              '<plan>',
              '  <plan_name>test_plan</plan_name>',
              '  <plan_steps>',
              '    <parallel>',
              '      <plan_step>',
              '        <step_name>step1</step_name>',
              '        <step_intent>intent1</step_intent>',
              '        <system_prompt></system_prompt>',
              '        <model_api>api1</model_api>',
              "        <input_parameters>['param1']</input_parameters>",
              '        <output>output1</output>',
              '        <reasoning>reasoning1</reasoning>',
              '        <iterations>1</iterations>',
              '        <is_list_output>False</is_list_output>',
              '      </plan_step>',
              '      <plan_step>',
              '        <step_name>step2</step_name>',
              '        <step_intent>intent2</step_intent>',
              '        <system_prompt></system_prompt>',
              '        <model_api>api2</model_api>',
              "        <input_parameters>['param2']</input_parameters>",
              '        <output>output2</output>',
              '        <reasoning>reasoning2</reasoning>',
              '        <iterations>1</iterations>',
              '        <is_list_output>False</is_list_output>',
              '      </plan_step>',
              '    </parallel>',
              '  </plan_steps>',
              '</plan>',
          ],
      ),
      (
          'mixed_sequential_and_parallel',
          [
              opal_plan_step.OpalPlanStep(
                  step_name='step1',
                  step_intent='intent1',
                  model_api='api1',
                  input_parameters=['param1'],
                  output='output1',
                  reasoning='reasoning1',
              ),
              [
                  opal_plan_step.OpalPlanStep(
                      step_name='step2',
                      step_intent='intent2',
                      model_api='api2',
                      input_parameters=['param2'],
                      output='output2',
                      reasoning='reasoning2',
                  ),
                  opal_plan_step.OpalPlanStep(
                      step_name='step3',
                      step_intent='intent3',
                      model_api='api3',
                      input_parameters=['param3'],
                      output='output3',
                      reasoning='reasoning3',
                  ),
              ],
              opal_plan_step.OpalPlanStep(
                  step_name='step4',
                  step_intent='intent4',
                  model_api='api4',
                  input_parameters=['param4'],
                  output='output4',
                  reasoning='reasoning4',
              ),
          ],
          [
              '<plan>',
              '  <plan_name>test_plan</plan_name>',
              '  <plan_steps>',
              '    <plan_step>',
              '      <step_name>step1</step_name>',
              '      <step_intent>intent1</step_intent>',
              '      <system_prompt></system_prompt>',
              '      <model_api>api1</model_api>',
              "      <input_parameters>['param1']</input_parameters>",
              '      <output>output1</output>',
              '      <reasoning>reasoning1</reasoning>',
              '      <iterations>1</iterations>',
              '      <is_list_output>False</is_list_output>',
              '    </plan_step>',
              '    <parallel>',
              '      <plan_step>',
              '        <step_name>step2</step_name>',
              '        <step_intent>intent2</step_intent>',
              '        <system_prompt></system_prompt>',
              '        <model_api>api2</model_api>',
              "        <input_parameters>['param2']</input_parameters>",
              '        <output>output2</output>',
              '        <reasoning>reasoning2</reasoning>',
              '        <iterations>1</iterations>',
              '        <is_list_output>False</is_list_output>',
              '      </plan_step>',
              '      <plan_step>',
              '        <step_name>step3</step_name>',
              '        <step_intent>intent3</step_intent>',
              '        <system_prompt></system_prompt>',
              '        <model_api>api3</model_api>',
              "        <input_parameters>['param3']</input_parameters>",
              '        <output>output3</output>',
              '        <reasoning>reasoning3</reasoning>',
              '        <iterations>1</iterations>',
              '        <is_list_output>False</is_list_output>',
              '      </plan_step>',
              '    </parallel>',
              '    <plan_step>',
              '      <step_name>step4</step_name>',
              '      <step_intent>intent4</step_intent>',
              '      <system_prompt></system_prompt>',
              '      <model_api>api4</model_api>',
              "      <input_parameters>['param4']</input_parameters>",
              '      <output>output4</output>',
              '      <reasoning>reasoning4</reasoning>',
              '      <iterations>1</iterations>',
              '      <is_list_output>False</is_list_output>',
              '    </plan_step>',
              '  </plan_steps>',
              '</plan>',
          ],
      ),
  )
  def test_render(self, plan_steps, expected_substrings):
    plan = opal_plan.OpalPlan(
        plan_name='test_plan',
        plan_steps=plan_steps,
    )
    rendered = plan.render
    self.assertCountEqual(rendered.splitlines(), expected_substrings)


if __name__ == '__main__':
  absltest.main()
