"""Tests for tool_utils."""

from absl.testing import absltest
from absl.testing import parameterized
from opal_adk.tools.chat import chat_request_user_input
from opal_adk.tools.chat import instructions as chat_instructions
from opal_adk.tools.generate import generate_text
from opal_adk.tools.generate import instructions as generate_instructions
from opal_adk.tools.system import instructions as system_instructions
from opal_adk.tools.system import objective_failed
from opal_adk.tools.system import objective_fulfilled
from opal_adk.types import model_constraint
from opal_adk.util import tool_utils

UIType = tool_utils.UIType


class ToolUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('unspecified', model_constraint.ModelConstraint.UNSPECIFIED),
      ('text_flash', model_constraint.ModelConstraint.TEXT_FLASH),
      ('text_pro', model_constraint.ModelConstraint.TEXT_PRO),
      ('image', model_constraint.ModelConstraint.IMAGE),
      ('video', model_constraint.ModelConstraint.VIDEO),
      ('speech', model_constraint.ModelConstraint.SPEECH),
      ('music', model_constraint.ModelConstraint.MUSIC),
  )
  def test_get_tools_with_model_constraints(self, constraint):
    tools = tool_utils.get_tools_with_model_constraints(constraint)
    expected_tools = [
        (
            system_instructions.SYSTEM_FUNCTIONS_INSTRUCTIONS,
            [
                objective_failed.objective_failed,
                objective_fulfilled.objective_fulfilled,
            ],
        ),
        (
            generate_instructions.GENERATE_INSTRUCTIONS,
            [generate_text.generate_text],
        ),
    ]
    self.assertEqual(tools, expected_tools)

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
