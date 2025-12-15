"""Tests for executor."""

import unittest
from unittest import mock

from google.genai import types
from opal_adk.data_model import opal_plan_step
from opal_adk.execution import executor


class ExecutorTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()

    self.mock_session_service_patcher = mock.patch.object(
        executor.in_memory_session_service,
        'InMemorySessionService',
        autospec=True,
    )
    self.mock_session_service_cls = self.mock_session_service_patcher.start()
    self.addCleanup(self.mock_session_service_patcher.stop)

    self.mock_memory_service_patcher = mock.patch.object(
        executor.in_memory_memory_service,
        'InMemoryMemoryService',
        autospec=True,
    )
    self.mock_memory_service_cls = self.mock_memory_service_patcher.start()
    self.addCleanup(self.mock_memory_service_patcher.stop)

    self.mock_runner_patcher = mock.patch.object(
        executor.runners, 'Runner', autospec=True
    )
    self.mock_runner_cls = self.mock_runner_patcher.start()
    self.addCleanup(self.mock_runner_patcher.stop)

    self.mock_sequential_agent_patcher = mock.patch.object(
        executor.sequential_agent, 'SequentialAgent', autospec=True
    )
    self.mock_sequential_agent_cls = self.mock_sequential_agent_patcher.start()
    self.addCleanup(self.mock_sequential_agent_patcher.stop)

    self.mock_research_agent_patcher = mock.patch.object(
        executor, 'research_agent', autospec=True
    )
    self.mock_research_agent_mod = self.mock_research_agent_patcher.start()
    self.addCleanup(self.mock_research_agent_patcher.stop)

    self.mock_report_writing_agent_patcher = mock.patch.object(
        executor, 'report_writing_agent', autospec=True
    )
    self.mock_report_writing_agent_mod = (
        self.mock_report_writing_agent_patcher.start()
    )
    self.addCleanup(self.mock_report_writing_agent_patcher.stop)

    self.executor = executor.AgentExecutor()

  def test_create_content_from_string(self):
    content = self.executor._create_content_from_string('hello')
    expected_content = types.Content(
        role='user', parts=[types.Part(text='hello')]
    )
    self.assertEqual(content, expected_content)

  async def test_execute_deep_research_agent(self):
    user_id = 'test_user'
    step = opal_plan_step.OpalPlanStep(
        step_name='test_step',
        step_intent='do research',
        model_api='gemini',
        iterations=3,
    )

    # Assuming 'Session' is the class returned by create_session.
    # Replace 'opal_adk.data_model.Session' with the actual class path.
    mock_session = mock.AsyncMock()
    mock_session.id = 'session_123'
    self.executor.session_service.create_session = mock.AsyncMock(
        return_value=mock_session
    )

    mock_runner_instance = self.mock_runner_cls.return_value
    mock_runner_instance.run_async = mock.Mock(
        return_value='async_generator_result'
    )

    result = await self.executor.execute_deep_research_agent(user_id, step)

    self.mock_research_agent_mod.deep_research_agent.assert_called_once_with(
        iterations=3
    )
    self.mock_report_writing_agent_mod.report_writing_agent.assert_called_once()

    self.mock_sequential_agent_cls.assert_called_once()
    call_args = self.mock_sequential_agent_cls.call_args
    self.assertEqual(call_args.kwargs['name'], 'deep_research_agent')
    self.assertEqual(len(call_args.kwargs['sub_agents']), 2)

    self.executor.session_service.create_session.assert_called_once_with(
        app_name='test_step', user_id='test_user'
    )

    self.mock_runner_cls.assert_called_once_with(
        app_name='test_step',
        agent=self.mock_sequential_agent_cls.return_value,
        session_service=self.executor.session_service,
        memory_service=self.executor.memory_service,
    )

    mock_runner_instance.run_async.assert_called_once()
    run_args = mock_runner_instance.run_async.call_args
    self.assertEqual(run_args.kwargs['user_id'], 'test_user')
    self.assertEqual(run_args.kwargs['session_id'], 'session_123')

    expected_content = types.Content(
        role='user', parts=[types.Part(text='do research')]
    )
    actual_content = run_args.kwargs['new_message']
    self.assertEqual(actual_content, expected_content)

    self.assertEqual(result, 'async_generator_result')


if __name__ == '__main__':
  unittest.main()
