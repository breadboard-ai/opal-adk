from unittest import mock

from absl.testing import parameterized
from google.adk.agents import llm_agent
from google.adk.planners import built_in_planner
from opal_adk.agents import node_agent
from opal_adk.types import models
from opal_adk.types import ui_type as opal_adk_ui_types
from opal_adk.util import tool_utils

from google3.testing.pybase import googletest


class NodeAgentTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_get_tools = mock.patch.object(tool_utils, "get_tools").start()
    self.mock_get_tools_for_ui_type = mock.patch.object(
        tool_utils, "get_tools_for_ui_type"
    ).start()
    self.mock_llm_agent_cls = mock.patch.object(llm_agent, "LlmAgent").start()
    self.mock_planner_cls = mock.patch.object(
        built_in_planner, "BuiltInPlanner"
    ).start()
    self.addCleanup(mock.patch.stopall)

  def test_node_agent_initialization(self):
    self.mock_get_tools.return_value = []
    self.mock_get_tools_for_ui_type.return_value = []

    node_agent.node_agent(opal_adk_ui_types.UIType.CHAT)

    self.mock_get_tools.assert_called_once()
    self.mock_get_tools_for_ui_type.assert_called_once_with(
        ui_type=opal_adk_ui_types.UIType.CHAT
    )

  def test_node_agent_initialization_default_model(self):
    self.mock_get_tools.return_value = []
    self.mock_get_tools_for_ui_type.return_value = []

    node_agent.node_agent(
        opal_adk_ui_types.UIType.CHAT,
    )

    self.mock_llm_agent_cls.assert_called_once()
    _, kwargs = self.mock_llm_agent_cls.call_args
    self.assertEqual(kwargs["model"], models.Models.GEMINI_3_FLASH.value)

  def test_node_agent_initialization_custom_model(self):
    self.mock_get_tools.return_value = []
    self.mock_get_tools_for_ui_type.return_value = []
    custom_model = models.Models.GEMINI_2_5_PRO

    node_agent.node_agent(
        opal_adk_ui_types.UIType.CHAT,
        agent_model=custom_model,
    )

    self.mock_llm_agent_cls.assert_called_once()
    _, kwargs = self.mock_llm_agent_cls.call_args
    self.assertEqual(kwargs["model"], custom_model.value)

  def test_node_agent_aggregates_tools_and_instructions(self):
    self.mock_get_tools.return_value = [
        ("instruction1", ["tool1"]),
        ("instruction2", ["tool2", "tool3"]),
    ]
    self.mock_get_tools_for_ui_type.return_value = [
        ("ui_instruction", ["ui_tool"]),
    ]

    node_agent.node_agent(
        opal_adk_ui_types.UIType.CHAT,
    )

    self.mock_llm_agent_cls.assert_called_once()
    _, kwargs = self.mock_llm_agent_cls.call_args

    self.assertEqual(
        kwargs["static_instruction"],
        "instruction1\ninstruction2\nui_instruction",
    )
    self.assertEqual(kwargs["tools"], ["tool1", "tool2", "tool3", "ui_tool"])
    self.assertEqual(kwargs["name"], "opal_adk_node_agent")
    self.assertEqual(kwargs["output_key"], "opal_adk_node_agent_output")
    self.assertIn("Iteratively works", kwargs["description"])
    self.assertEqual(kwargs["generate_content_config"].temperature, 1.0)
    self.assertEqual(kwargs["generate_content_config"].top_p, 1)

    # Verify planner configuration
    self.mock_planner_cls.assert_called_once()
    _, planner_kwargs = self.mock_planner_cls.call_args
    thinking_config = planner_kwargs.get("thinking_config")
    self.assertTrue(thinking_config.include_thoughts)
    self.assertEqual(kwargs["planner"], self.mock_planner_cls.return_value)


if __name__ == "__main__":
  googletest.main()
