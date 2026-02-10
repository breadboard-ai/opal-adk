"""Tests for tool_utils."""

from absl.testing import absltest
from absl.testing import parameterized
from opal_adk.tools.chat import chat_request_user_input
from opal_adk.tools.chat import instructions as chat_instructions
from opal_adk.util import tool_utils

UIType = tool_utils.UIType


class ToolUtilsTest(parameterized.TestCase):

  def test_get_tools_for_ui_type_chat(self):
    tools = tool_utils.get_tools_for_ui_type(UIType.CHAT)
    expected_tools = [(
        chat_instructions.CHAT_INSTRUCTIONS,
        [chat_request_user_input.chat_request_user_input],
    )]
    self.assertEqual(tools, expected_tools)

  def test_get_tools_for_ui_type_unspecified(self):
    tool_utils.get_tools_for_ui_type(UIType.A2UI)
    self.assertRaisesRegex(NotImplementedError('tool_utils: *'))

  def test_get_tools_for_ui_type_unspecified(self):
    tools = tool_utils.get_tools_for_ui_type(UIType.UNSPECIFIED)
    self.assertEqual(tools, [])


if __name__ == '__main__':
  absltest.main()
