"""Unit tests for deep_research_agent_workflow."""

import unittest
from unittest import mock

from google.adk.agents import sequential_agent
from opal_adk import models
from opal_adk.agents import report_writing_agent
from opal_adk.agents import research_agent
from opal_adk.workflows import deep_research_agent_workflow


class DeepResearchAgentWorkflowTest(unittest.TestCase):
  """Unit tests for the deep_research_agent_workflow."""

  @mock.patch.object(sequential_agent, 'SequentialAgent', autospec=True)
  @mock.patch.object(
      report_writing_agent, 'report_writing_agent', autospec=True
  )
  @mock.patch.object(research_agent, 'deep_research_agent', autospec=True)
  def test_deep_research_agent_workflow_creates_sequential_agent(
      self,
      mock_research_agent,
      mock_report_writing_agent,
      mock_sequential_agent,
  ):
    """Tests that deep_research_agent_workflow creates a SequentialAgent with the correct parameters."""
    mock_research_agent_instance = mock.Mock()
    mock_report_writing_agent_instance = mock.Mock()
    mock_research_agent.return_value = mock_research_agent_instance
    mock_report_writing_agent.return_value = mock_report_writing_agent_instance

    workflow = deep_research_agent_workflow.deep_research_agent_workflow()

    mock_research_agent.assert_called_once_with(
        model=models.Models.GEMINI_2_5_FLASH
    )
    mock_report_writing_agent.assert_called_once_with(
        model=models.Models.GEMINI_2_5_FLASH,
        parent_agent_output_key=research_agent.OUTPUT_KEY,
    )
    mock_sequential_agent.assert_called_once_with(
        name=deep_research_agent_workflow.WORKFLOW_NAME,
        description=(
            'A multi-agent sequential flow that performs research based on a'
            ' user query and then generates a cited research report.'
        ),
        sub_agents=[
            mock_research_agent_instance,
            mock_report_writing_agent_instance,
        ],
    )
    self.assertEqual(workflow, mock_sequential_agent.return_value)


if __name__ == '__main__':
  unittest.main()
