"""Tests for executor."""

import os
import unittest
from unittest import mock

from absl.testing import absltest
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

    self.mock_env_patcher = mock.patch(
        'opal_adk.error_handling.opal_adk_error.environment_util.is_prod_environment',
        return_value=True,
    )
    self.mock_env = self.mock_env_patcher.start()
    self.addCleanup(self.mock_env_patcher.stop)

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

    mock_session = mock.AsyncMock()
    mock_session.id = 'session_123'

    self.executor.session_service.get_session = mock.AsyncMock(
        return_value=None
    )
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

  async def test_execute_deep_research_agent_with_session_id(self):
    user_id = 'test_user'
    step = opal_plan_step.OpalPlanStep(
        step_name='test_step',
        step_intent='do research',
        model_api='gemini',
        iterations=3,
        input_parameters=['test_input'],
    )

    mock_session = mock.AsyncMock()
    mock_session.id = 'session_123'
    self.executor.session_service.get_session = mock.AsyncMock(
        return_value=mock_session
    )
    self.executor.session_service.create_session = mock.AsyncMock()

    mock_runner_instance = self.mock_runner_cls.return_value
    mock_runner_instance.run_async = mock.Mock(
        return_value='async_generator_result'
    )

    input_content = types.Content(
        role='user', parts=[types.Part(text='do research')]
    )
    result = await self.executor.execute_deep_research_agent(
        user_id,
        step,
        execution_inputs={'test_input': input_content},
        session_id='session_123',
    )

    self.executor.session_service.get_session.assert_called_once_with(
        app_name='test_step', user_id='test_user', session_id='session_123'
    )
    self.executor.session_service.create_session.assert_not_called()

    run_args = mock_runner_instance.run_async.call_args
    self.assertEqual(run_args.kwargs['session_id'], 'session_123')
    self.assertEqual(result, 'async_generator_result')

  async def test_populate_session_artifacts_success(self):
    mock_artifact_service = mock.AsyncMock()
    content = types.Content(
        role='user', parts=[types.Part(text='test content')]
    )
    execution_inputs = {'param1': content}
    mock_session = mock.MagicMock()
    mock_session.id = 'session_123'
    mock_session.state = {}

    await executor._populate_session_artifacts(
        app='test_app',
        user_id='test_user',
        artifact_service=mock_artifact_service,
        execution_inputs=execution_inputs,
        session=mock_session,
    )

    mock_artifact_service.save_artifact.assert_called_once_with(
        app_name='test_app',
        user_id='test_user',
        filename='param1',
        artifact=content.parts[0],
        session_id='session_123',
    )
    self.assertEqual(
        mock_session.state['saved_file_to_artifact_service'], 'param1'
    )
    self.assertEqual(mock_session.state['artifact_provided_by'], 'user')

  async def test_execute_deep_research_agent_missing_parameter(self):
    user_id = 'test_user'
    step = opal_plan_step.OpalPlanStep(
        step_name='test_step',
        step_intent='do research',
        model_api='gemini',
        iterations=3,
        input_parameters=['test_input'],
    )
    with self.assertRaisesRegex(
        ValueError, "Input parameter 'test_input' not found in execution inputs"
    ):
      await self.executor.execute_deep_research_agent(
          user_id, step, execution_inputs={'other_input': types.Content()}
      )

  async def test_populate_session_artifacts_empty_content_parts(self):
    mock_artifact_service = mock.AsyncMock()
    execution_inputs = {'param1': types.Content(role='user', parts=[])}
    mock_session = mock.MagicMock()
    mock_session.id = 'session_123'
    mock_session.state = {}

    with self.assertRaisesRegex(
        executor.opal_adk_error.OpalAdkError,
        "Input parameter 'param1' has no content in execution inputs",
    ):
      await executor._populate_session_artifacts(
          app='test_app',
          user_id='test_user',
          artifact_service=mock_artifact_service,
          execution_inputs=execution_inputs,
          session=mock_session,
      )

  @mock.patch.object(executor.node_agent, 'node_agent', autospec=True)
  @mock.patch.object(executor.loop_agent, 'LoopAgent', autospec=True)
  @mock.patch.object(
      executor.in_memory_artifact_service,
      'InMemoryArtifactService',
      autospec=True,
  )
  async def test_execute_agent_node(
      self,
      mock_artifact_service_cls,
      mock_loop_agent_cls,
      mock_node_agent_func,
  ):
    user_id = 'test_user'
    step = mock.MagicMock(spec=executor.agent_step.AgentStep)
    step.step_name = 'test_step'
    step.ui_type = executor.ui_type.UIType.CHAT
    step.input_parameters = ['param1']
    step.objective = 'test objective'
    step.invocation_id = 'inv_123'

    mock_session = mock.MagicMock()
    mock_session.id = 'session_123'
    mock_session.state = {}
    self.executor.session_service.get_session = mock.AsyncMock(
        return_value=mock_session
    )

    mock_runner_instance = self.mock_runner_cls.return_value
    mock_runner_instance.run_async = mock.Mock(
        return_value='async_generator_result'
    )

    input_content = types.Content(
        role='user', parts=[types.Part(text='test content')]
    )

    result = await self.executor.execute_agent_node(
        user_id=user_id,
        step=step,
        session_id='session_123',
        execution_inputs={'param1': input_content},
    )

    self.mock_runner_cls.assert_called_once_with(
        app_name='test_step',
        agent=mock_loop_agent_cls.return_value,
        session_service=self.executor.session_service,
        memory_service=self.executor.memory_service,
        artifact_service=mock_artifact_service_cls.return_value,
    )

    mock_artifact_service_instance = mock_artifact_service_cls.return_value
    mock_artifact_service_instance.save_artifact.assert_called_once_with(
        app_name='test_step',
        user_id='test_user',
        filename='param1',
        artifact=input_content.parts[0],
        session_id='session_123',
    )

    self.assertEqual(result, 'async_generator_result')

  async def test_execute_agent_node_missing_session_id_chat_ui(self):
    user_id = 'test_user'
    step = mock.MagicMock(spec=executor.agent_step.AgentStep)
    step.step_name = 'test_step'
    step.ui_type = executor.ui_type.UIType.CHAT

    with self.assertRaisesRegex(
        ValueError, 'Executor: session_id must be provided for chat UI type.'
    ):
      await self.executor.execute_agent_node(
          user_id=user_id,
          step=step,
          session_id=None,
      )

  def test_extract_input_parameter_success(self):
    step = opal_plan_step.OpalPlanStep(
        step_name='test_step',
        step_intent='do research',
        model_api='gemini',
        iterations=3,
        input_parameters=['test_input'],
    )
    result = executor._extract_input_parameter(step)
    self.assertEqual(result, 'test_input')

  def test_extract_input_parameter_raises_error(self):
    step = opal_plan_step.OpalPlanStep(
        step_name='test_step',
        step_intent='do research',
        model_api='gemini',
        iterations=3,
        input_parameters=['test_input1', 'test_input2'],
    )
    with self.assertRaisesRegex(
        ValueError, 'expected exactly one input parameter'
    ):
      executor._extract_input_parameter(step)


if __name__ == '__main__':
  absltest.main()
