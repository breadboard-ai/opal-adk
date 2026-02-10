"""Tests for report_writing_agent."""

from absl.testing import absltest
from absl.testing import parameterized
from google.adk.agents import llm_agent
from google.adk.planners import built_in_planner
from opal_adk.agents import report_writing_agent
from opal_adk.types import models


class ReportWritingAgentTest(parameterized.TestCase):

  @parameterized.parameters(list(models.Models))
  def test_report_writing_agent_creates_agent_with_model(self, model):
    agent = report_writing_agent.report_writing_agent(model=model)
    self.assertIsInstance(agent, llm_agent.LlmAgent)
    self.assertEqual(agent.model, model.value)
    self.assertEqual(agent.name, report_writing_agent.AGENT_NAME)
    self.assertEqual(
        agent.description,
        "Agent that takes input and produces a well cited report.",
    )
    self.assertEqual(
        agent.instruction,
        report_writing_agent.REPORT_WRITING_SYSTEM_INSTRUCTIONS,
    )
    self.assertEqual(agent.output_key, report_writing_agent.OUTPUT_KEY)
    self.assertIsInstance(agent.planner, built_in_planner.BuiltInPlanner)
    self.assertTrue(agent.planner.thinking_config.include_thoughts)

  def test_report_writing_agent_with_parent_agent_output_key(self):
    parent_key = "parent_output"
    agent = report_writing_agent.report_writing_agent(
        parent_agent_output_key=parent_key
    )
    expected_instructions = (
        report_writing_agent.REPORT_WRITING_SYSTEM_INSTRUCTIONS
        + report_writing_agent.previous_agent_output_instructions(parent_key)
    )
    self.assertEqual(agent.instruction, expected_instructions)


if __name__ == "__main__":
  absltest.main()
