"""Tests for generate_text."""

import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from opal_adk.clients import vertex_ai_client
from opal_adk.error_handling import opal_adk_error
from opal_adk.tools import fetch_url_contents_tool
from opal_adk.tools import map_search_tool
from opal_adk.tools import vertex_search_tool
from opal_adk.tools.generate import generate_text
from opal_adk.types import models
from opal_adk.types import output_type


class GenerateTextTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def setUp(self):
    super().setUp()
    # Patcher for vertex_search_tool.search_agent_tool
    self.search_tool_patcher = mock.patch.object(
        vertex_search_tool, 'search_agent_tool'
    )
    self.mock_search_tool = self.search_tool_patcher.start()

    # Patcher for map_search_tool.MapSearchTool
    self.map_tool_patcher = mock.patch.object(map_search_tool, 'MapSearchTool')
    self.mock_map_tool = self.map_tool_patcher.start()

    # Patcher for fetch_url_contents_tool.FetchUrlContentsTool
    self.fetch_url_tool_patcher = mock.patch.object(
        fetch_url_contents_tool, 'FetchUrlContentsTool'
    )
    self.mock_fetch_url_tool = self.fetch_url_tool_patcher.start()

    # Patcher for models.simple_model_to_model
    self.model_converter_patcher = mock.patch.object(
        models, 'simple_model_to_model', autospec=True
    )
    self.mock_model_converter = self.model_converter_patcher.start()
    # Mock the return value of simple_model_to_model to have a .value attribute
    self.mock_model_converter.return_value.value = 'mock-model-value'

    # Patcher for vertex_ai_client.create_vertex_ai_client
    self.client_patcher = mock.patch.object(
        vertex_ai_client, 'create_vertex_ai_client', autospec=True
    )
    self.mock_create_client = self.client_patcher.start()
    self.mock_client = self.mock_create_client.return_value
    # Mock aio.models.generate_content
    self.mock_client.aio.models.generate_content = mock.AsyncMock()

  def tearDown(self):
    self.search_tool_patcher.stop()
    self.map_tool_patcher.stop()
    self.fetch_url_tool_patcher.stop()
    self.model_converter_patcher.stop()
    self.client_patcher.stop()
    super().tearDown()

  async def test_generate_text_default_args(self):
    instructions = 'Test instructions'
    response = await generate_text.generate_text(instructions)

    self.mock_model_converter.assert_called_once_with(models.SimpleModel.FLASH)
    self.mock_create_client.assert_called_once()

    self.mock_client.aio.models.generate_content.assert_called_once()
    _, kwargs = self.mock_client.aio.models.generate_content.call_args

    self.assertEqual(kwargs['model'], 'mock-model-value')

    # Compare contents carefully or just check structure if equality is hard
    # google.genai.types usually support equality check
    # But let's check the structure to be safe if __eq__ is not
    # implemented as expected
    actual_content = kwargs['contents']
    self.assertEqual(actual_content.role, 'user')
    self.assertEqual(actual_content.parts[0].text, instructions)

    # Check config
    config = kwargs['config']
    self.assertEqual(config.tools, [])
    self.assertEqual(config.system_instruction.role, 'user')
    self.assertIn('no chit-chat', config.system_instruction.parts[0].text)

    self.assertEqual(
        response, self.mock_client.aio.models.generate_content.return_value
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='invalid_model',
          kwargs={'model': 'invalid-model'},
          error_msg='Invalid model: invalid-model',
      ),
      dict(
          testcase_name='invalid_output_format',
          kwargs={'output_format': 'invalid-format'},
          error_msg='Invalid output_format: invalid-format',
      ),
  )
  async def test_generate_text_invalid_args(self, kwargs, error_msg):
    with self.assertRaisesRegex(opal_adk_error.OpalAdkError, error_msg):
      await generate_text.generate_text('instructions', **kwargs)

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
          check_return_value=True,
      ),
      dict(
          testcase_name='url_context',
          kwargs={'url_context': True},
          tool_attr_name='mock_fetch_url_tool',
          check_return_value=True,
      ),
  )
  async def test_generate_text_with_grounding(
      self, kwargs, tool_attr_name, check_return_value
  ):
    await generate_text.generate_text('instructions', **kwargs)

    mock_tool = getattr(self, tool_attr_name)
    if check_return_value:
      mock_tool.assert_called_once()
      expected_tool = mock_tool.return_value
    else:
      expected_tool = mock_tool

    _, call_kwargs = self.mock_client.aio.models.generate_content.call_args
    config = call_kwargs['config']
    self.assertIn(expected_tool, config.tools)

  async def test_generate_text_all_options(self):
    await generate_text.generate_text(
        'instructions',
        model=models.SimpleModel.PRO.value,
        output_format=output_type.OutputType.FILE.value,
        search_grounding=True,
        maps_grounding=True,
        url_context=True,
    )

    self.mock_model_converter.assert_called_once_with(models.SimpleModel.PRO)
    _, kwargs = self.mock_client.aio.models.generate_content.call_args
    tools = kwargs['config'].tools
    self.assertIn(self.mock_search_tool.return_value, tools)
    self.assertIn(self.mock_map_tool.return_value, tools)
    self.assertIn(self.mock_fetch_url_tool.return_value, tools)
    self.assertLen(tools, 3)


if __name__ == '__main__':
  absltest.main()
