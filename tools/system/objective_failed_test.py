"""Tests for objective_failed."""

import unittest
from unittest import mock
from opal_adk.tools.system import objective_failed


class ObjectiveFailedTest(unittest.TestCase):

  def test_objective_failed(self):
    mock_tool_context = mock.Mock()
    result = objective_failed.objective_failed(mock_tool_context)

    self.assertEqual(result, {"status": "failed"})
    self.assertTrue(mock_tool_context.actions.escalate)
    self.assertTrue(mock_tool_context.actions.skip_summarization)


if __name__ == "__main__":
  unittest.main()
