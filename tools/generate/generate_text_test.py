"""Tests for generate_text."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from google.adk.agents import llm_agent
from google.adk.tools import agent_tool
from opal_adk.tools import fetch_url_contents_tool
from opal_adk.tools import map_search_tool
from opal_adk.tools import vertex_search_tool
from opal_adk.tools.generate import generate_text
from opal_adk.types import models
from opal_adk.types import output_type


class GenerateTextTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Patcher for vertex_search_tool.search_agent_tool
    self.search_tool_patcher = mock.patch.object(
        vertex_search_tool, 'search_agent_tool' 
    )
    self.mock_search_tool = self.search_tool_patcher.start()

    # Patcher for map_search_tool.MapSearchTool
    self.map_tool_patcher = mock.patch.object(
        map_search_tool, 'MapSearchTool', autospec=True
    )
    self.mock_map_tool = self.map_tool_patcher.start()

    # Patcher for fetch_url_contents_tool.FetchUrlContentsTool
    self.fetch_url_tool_patcher = mock.patch.object(
        fetch_url_contents_tool, 'FetchUrlContentsTool', autospec=True
    )
    self.mock_fetch_url_tool = self.fetch_url_tool_patcher.start()

    # Patcher for agent_tool.AgentTool
    self.agent_tool_patcher = mock.patch.object(
        agent_tool, 'AgentTool', autospec=True
    )
    self.mock_agent_tool_cls = self.agent_tool_patcher.start()

    # Patcher for llm_agent.LlmAgent
    self.llm_agent_patcher = mock.patch.object(
        llm_agent, 'LlmAgent', autospec=True
    )
    self.mock_llm_agent_cls = self.llm_agent_patcher.start()

    # Patcher for models.simple_model_to_model
    self.model_converter_patcher = mock.patch.object(
        models, 'simple_model_to_model', autospec=True
    )
    self.mock_model_converter = self.model_converter_patcher.start()

  def tearDown(self):
    self.search_tool_patcher.stop()
    self.map_tool_patcher.stop()
    self.fetch_url_tool_patcher.stop()
    self.agent_tool_patcher.stop()
    self.llm_agent_patcher.stop()
    self.model_converter_patcher.stop()
    super().tearDown()

  def test_generate_text_default_args(self):
    tool = generate_text.generate_text()

    self.mock_model_converter.assert_called_once_with(models.SimpleModel.FLASH)
    self.mock_llm_agent_cls.assert_called_once()
    _, kwargs = self.mock_llm_agent_cls.call_args
    self.assertEqual(kwargs['name'], 'generate_text agent as a tool')
    self.assertEqual(kwargs['model'], self.mock_model_converter.return_value)
    self.assertEqual(kwargs['tools'], [])
    self.mock_agent_tool_cls.assert_called_once_with(
        self.mock_llm_agent_cls.return_value
    )
    self.assertEqual(tool, self.mock_agent_tool_cls.return_value)

  @parameterized.named_parameters(
      dict(
          testcase_name='search_grounding',
          kwargs={'search_grounding': True},
          tool_attr_name='mock_search_tool',
          check_return_value=True,
      ),
      dict(
          testcase_name='maps_grounding',
          kwargs={'maps_grounding': True},
          tool_attr_name='mock_map_tool',
          check_return_value=False,
      ),
      dict(
          testcase_name='url_context',
          kwargs={'url_context': True},
          tool_attr_name='mock_fetch_url_tool',
          check_return_value=False,
      ),
  )
  def test_generate_text_with_grounding(
      self, kwargs, tool_attr_name, check_return_value
  ):
    generate_text.generate_text(**kwargs)

    mock_tool = getattr(self, tool_attr_name)
    if check_return_value:
      mock_tool.assert_called_once()
      expected_tool = mock_tool.return_value
    else:
      expected_tool = mock_tool

    _, call_kwargs = self.mock_llm_agent_cls.call_args
    self.assertIn(expected_tool, call_kwargs['tools'])

  def test_generate_text_all_options(self):
    generate_text.generate_text(
        model=models.SimpleModel.PRO,
        output_format=output_type.OutputType.FILE,
        file_name='test_file',
        search_grounding=True,
        maps_grounding=True,
        url_context=True,
    )

    self.mock_model_converter.assert_called_once_with(models.SimpleModel.PRO)
    _, kwargs = self.mock_llm_agent_cls.call_args
    tools = kwargs['tools']
    self.assertIn(self.mock_search_tool.return_value, tools)
    self.assertIn(self.mock_map_tool, tools)
    self.assertIn(self.mock_fetch_url_tool, tools)
    self.assertLen(tools, 3)


if __name__ == '__main__':
  absltest.main()
