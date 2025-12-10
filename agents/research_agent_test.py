"""Tests for research_agent."""

import unittest
from unittest import mock

from absl.testing import parameterized
from opal_adk.agents import research_agent
from opal_adk.tools import fetch_url_contents_tool
from opal_adk.tools import map_search_tool
from opal_adk.tools import vertex_search_tool


def dummy_tool():
  pass


class ResearchAgentTest(parameterized.TestCase):
  """Tests for research_agent."""

  def setUp(self):
    super().setUp()
    patcher = mock.patch('google.genai.Client')
    patcher.start()
    self.addCleanup(patcher.stop)

    # Mock vertex_ai_client.create_vertex_ai_client to avoid flag usage
    client_patcher = mock.patch(
        'opal_adk.agents.research_agent.vertex_ai_client.create_vertex_ai_client'
    )
    self.mock_create_client = client_patcher.start()
    self.addCleanup(client_patcher.stop)
    self.mock_create_client.return_value = mock.MagicMock()

  @parameterized.named_parameters(
      ('first_iteration', True, 'the first step', 'the next step'),
      ('next_iteration', False, 'the next step', 'the first step'),
  )
  def test_research_system_instructions(
      self, is_first_iteration, expected_in, expected_not_in
  ):
    """Tests that research_system_instructions returns correct string."""
    agent = research_agent.deep_research_agent(
        is_first_iteration=is_first_iteration
    )
    instructions = agent.sub_agents[0].instruction
    self.assertIn(expected_in, instructions)
    self.assertNotIn(expected_not_in, instructions)

  def test_deep_research_agent_creates_agent_with_defaults(self):
    """Tests that deep_research_agent creates an agent with default parameters."""
    agent = research_agent.deep_research_agent(is_first_iteration=True)
    self.assertEqual(agent.name, 'research_agent_orchestrator')
    self.assertLen(agent.sub_agents, 1)
    sub_agent = agent.sub_agents[0]
    self.assertEqual(sub_agent.name, research_agent.AGENT_NAME)
    self.assertEqual(sub_agent.output_key, research_agent.OUTPUT_KEY)

    fetch_tools = [
        t
        for t in sub_agent.tools
        if isinstance(t, fetch_url_contents_tool.FetchUrlContentsTool)
    ]
    self.assertLen(fetch_tools, 1)
    self.assertLen(sub_agent.tools, 3)

    map_tools = [
        t
        for t in sub_agent.tools
        if isinstance(t, map_search_tool.MapSearchTool)
    ]
    self.assertLen(map_tools, 1)

    vertex_tools = [
        t
        for t in sub_agent.tools
        if isinstance(t, vertex_search_tool.VertexSearchTool)
    ]
    self.assertLen(vertex_tools, 1)

    self.assertEqual(
        sub_agent.instruction,
        research_agent.research_system_instructions(True),
    )

  def test_deep_research_agent_with_additional_tools(self):
    """Tests that deep_research_agent includes additional tools."""
    agent = research_agent.deep_research_agent(
        is_first_iteration=True,
        additional_tools=[dummy_tool],
    )
    sub_agent = agent.sub_agents[0]
    self.assertLen(sub_agent.tools, 4)
    self.assertIn(dummy_tool, sub_agent.tools)

    fetch_tools = [
        t
        for t in sub_agent.tools
        if isinstance(t, fetch_url_contents_tool.FetchUrlContentsTool)
    ]
    self.assertLen(fetch_tools, 1)

    map_tools = [
        t
        for t in sub_agent.tools
        if isinstance(t, map_search_tool.MapSearchTool)
    ]
    self.assertLen(map_tools, 1)

    vertex_tools = [
        t
        for t in sub_agent.tools
        if isinstance(t, vertex_search_tool.VertexSearchTool)
    ]
    self.assertLen(vertex_tools, 1)

  def test_deep_research_agent_next_iteration_instruction(self):
    """Tests that deep_research_agent sets correct instruction for next iteration."""
    agent = research_agent.deep_research_agent(is_first_iteration=False)
    self.assertEqual(
        agent.sub_agents[0].instruction,
        research_agent.research_system_instructions(False),
    )

  def test_deep_research_agent_with_parent_agent_output_key(self):
    """Tests that deep_research_agent includes parent agent output instructions."""
    agent = research_agent.deep_research_agent(
        parent_agent_output_key='parent_key',
        is_first_iteration=False,
    )
    self.assertIn(
        research_agent.previous_agent_output_instructions('parent_key'),
        agent.sub_agents[0].instruction,
    )
    self.assertIn(
        research_agent.research_system_instructions(False),
        agent.sub_agents[0].instruction,
    )


if __name__ == '__main__':
  unittest.main()
