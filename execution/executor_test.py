"""Tests for executor."""

import os
import unittest
from unittest import mock

from absl.testing import parameterized
from google.genai import types
from opal_adk.data_model import opal_plan_step
from opal_adk.execution import executor


class ExecutorTest(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):

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

    self.mock_deep_research_workflow_patcher = mock.patch.object(
        executor, 'deep_research_agent_workflow', autospec=True
    )
    self.mock_deep_research_workflow = (
        self.mock_deep_research_workflow_patcher.start()
    )
    self.addCleanup(self.mock_deep_research_workflow_patcher.stop)
    self.env_patcher = mock.patch.dict(
        os.environ,
        {
            'GOOGLE_CLOUD_PROJECT': 'test_project',
            'GOOGLE_CLOUD_LOCATION': 'test_location',
            'GOOGLE_GENAI_USE_VERTEXAI': 'true',
        },
    )
    self.env_patcher.start()
    self.addCleanup(self.env_patcher.stop)

    self.executor = executor.AgentExecutor()

  @parameterized.named_parameters(
      (
          'missing_project_id',
          {'location': 'us-central1'},
          (
              'Both project_id and location must be provided, but got'
              " project_id=None and location='us-central1'"
          ),
      ),
      (
          'missing_location',
          {'project_id': 'my-project'},
          (
              'Both project_id and location must be provided, but got'
              " project_id='my-project' and location=None"
          ),
      ),
      (
          'missing_env_vars',
          {},
          (
              'When project_id and location are not provided, the following'
              ' environment variables must be set: GOOGLE_CLOUD_PROJECT,'
              ' GOOGLE_CLOUD_LOCATION, GOOGLE_GENAI_USE_VERTEXAI'
          ),
      ),
  )
  def test_init_raises_error(self, kwargs, error_message):
    with mock.patch.dict(os.environ, clear=True):
      with self.assertRaisesRegex(ValueError, error_message):
        executor.AgentExecutor(**kwargs)

  @parameterized.named_parameters(
      (
          'vertex_ai',
          {'project_id': 'p', 'location': 'l'},
          {
              'GOOGLE_CLOUD_PROJECT': 'p',
              'GOOGLE_CLOUD_LOCATION': 'l',
              'GOOGLE_GENAI_USE_VERTEXAI': 'true',
          },
      ),
      (
          'api_key',
          {'genai_api_key': 'test_key'},
          {
              'GOOGLE_API_KEY': 'test_key',
              'GOOGLE_GENAI_USE_VERTEXAI': 'false',
          },
      ),
  )
  def test_init_sets_env_vars(self, kwargs, expected_env):
    with mock.patch.dict(os.environ, clear=True):
      executor.AgentExecutor(**kwargs)
      for key, value in expected_env.items():
        self.assertEqual(os.environ[key], value)

  def test_create_content_from_string(self):
    content = executor._create_content_from_string('hello')
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
        input_parameters=['test_input'],
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

    input_content = types.Content(
        role='user', parts=[types.Part(text='do research')]
    )
    result = await self.executor.execute_deep_research_agent(
        user_id, step, execution_inputs={'test_input': input_content}
    )

    self.mock_deep_research_workflow.deep_research_agent_workflow.assert_called_once_with(
        num_iterations=3
    )

    self.executor.session_service.create_session.assert_called_once_with(
        app_name='test_step', user_id='test_user', session_id=None
    )

    self.mock_runner_cls.assert_called_once_with(
        app_name='test_step',
        agent=self.mock_deep_research_workflow.deep_research_agent_workflow.return_value,
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
