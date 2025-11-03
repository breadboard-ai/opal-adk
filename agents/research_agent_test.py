"""Tests for research_agent."""

import unittest
from opal_adk.agents import research_agent
from opal_adk.tools import fetch_url_contents
from opal_adk.tools import map_search
from opal_adk.tools import vertex_search


def dummy_tool():
  pass


class ResearchAgentTest(unittest.TestCase):
  """Tests for research_agent."""

  def test_research_system_instructions_first_iteration(self):
    """Tests that research_system_instructions returns correct string for first iteration."""
    agent = research_agent.deep_research_agent(is_first_iteration=True)
    instructions = agent.instruction
    self.assertIn('the first step', instructions)
    self.assertNotIn('the next step', instructions)

  def test_research_system_instructions_next_iteration(self):
    """Tests that research_system_instructions returns correct string for next iteration."""
    agent = research_agent.deep_research_agent(is_first_iteration=False)
    instructions = agent.instruction
    self.assertIn('the next step', instructions)
    self.assertNotIn('the first step', instructions)

  def test_deep_research_agent_creates_agent_with_defaults(self):
    """Tests that deep_research_agent creates an agent with default parameters."""
    agent = research_agent.deep_research_agent(is_first_iteration=True)
    self.assertEqual(agent.name, research_agent.AGENT_NAME)
    self.assertEqual(agent.output_key, research_agent.OUTPUT_KEY)
    self.assertCountEqual(
        agent.tools,
        [
            map_search.search_google_maps_places,
            fetch_url_contents.fetch_url,
            vertex_search.search_via_vertex_ai,
        ],
    )
    self.assertEqual(
        agent.instruction, research_agent.research_system_instructions(True)
    )

  def test_deep_research_agent_with_additional_tools(self):
    """Tests that deep_research_agent includes additional tools."""
    agent = research_agent.deep_research_agent(
        is_first_iteration=True,
        additional_tools=[dummy_tool],
    )
    self.assertCountEqual(
        agent.tools,
        [
            map_search.search_google_maps_places,
            fetch_url_contents.fetch_url,
            vertex_search.search_via_vertex_ai,
            dummy_tool,
        ],
    )

  def test_deep_research_agent_next_iteration_instruction(self):
    """Tests that deep_research_agent sets correct instruction for next iteration."""
    agent = research_agent.deep_research_agent(is_first_iteration=False)
    self.assertEqual(
        agent.instruction, research_agent.research_system_instructions(False)
    )

  def test_deep_research_agent_with_parent_agent_output_key(self):
    """Tests that deep_research_agent includes parent agent output instructions."""
    agent = research_agent.deep_research_agent(
        parent_agent_output_key='parent_key',
        is_first_iteration=False,
    )
    self.assertIn(
        research_agent.previous_agent_output_instructions('parent_key'),
        agent.instruction,
    )
    self.assertIn(
        research_agent.research_system_instructions(False),
        agent.instruction,
    )


if __name__ == '__main__':
  unittest.main()
