"""Tests for objective_fulfilled."""

import unittest
from unittest import mock
from opal_adk.tools.system import objective_fulfilled


class ObjectiveFulfilledTest(unittest.TestCase):

  def test_objective_fulfilled(self):
    mock_tool_context = mock.Mock()

    result = objective_fulfilled.objective_fulfilled(
        mock_tool_context, objective_outcome="I did it!"
    )

    self.assertEqual(
        result, {"status": "success", "objective_outcome": "I did it!"}
    )
    self.assertTrue(mock_tool_context.actions.escalate)
    self.assertTrue(mock_tool_context.actions.skip_summarization)


if __name__ == "__main__":
  unittest.main()
