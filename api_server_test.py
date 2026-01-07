"""Tests for api_server."""

import unittest
from unittest import mock

import fastapi
from fastapi import testclient
from google.adk.events import event
from opal_adk import api_server
from opal_adk.execution import executor


class ApiServerTest(unittest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    # Create a new app instance for testing to ensure isolation and include
    # router because the router is included in main() in api_server.py.
    self.app = fastapi.FastAPI()
    self.app.include_router(api_server.router)

  @mock.patch.object(executor, "AgentExecutor", autospec=True)
  def test_execute_deep_research_agent(self, mock_executor_cls):
    with testclient.TestClient(self.app) as test_client:
      mock_executor_instance = mock_executor_cls.return_value

      async def actual_generator():
        mock_event1 = mock.create_autospec(event.Event, instance=True)
        mock_event1.model_dump_json.return_value = '"Step 1"'
        yield mock_event1
        mock_event2 = mock.create_autospec(event.Event, instance=True)
        mock_event2.model_dump_json.return_value = '"Step 2"'
        yield mock_event2

      async def mock_method(*args, **kwargs):
        return actual_generator()

      mock_executor_instance.execute_deep_research_agent.side_effect = (
          mock_method
      )

      request_data = {
          "model": "gemini-pro",
          "query": "research something",
          "agent_parameters": {"param": "value"},
          "user_id": "user123",
          "iterations": 2,
      }

      response = test_client.post(
          "/execute_deep_research_agent", json=request_data
      )

      self.assertEqual(response.status_code, 200)
      self.assertEqual(
          response.text, 'response: "Step 1"\n\nresponse: "Step 2"\n\n'
      )

      mock_executor_instance.execute_deep_research_agent.assert_called_once()
      call_kwargs = (
          mock_executor_instance.execute_deep_research_agent.call_args.kwargs
      )
      self.assertEqual(call_kwargs["user_id"], "user123")
      self.assertEqual(
          call_kwargs["opal_step"].step_intent, "research something"
      )
      self.assertEqual(call_kwargs["opal_step"].model_api, "gemini-pro")

  @mock.patch.object(executor, "AgentExecutor")
  def test_execute_deep_research_agent_invalid_request(self, mock_executor_cls):
    request_data = {
        "model": "gemini-pro",
        # Missing query
        "agent_parameters": {},
        "user_id": "user123",
        "iterations": 2,
    }

    with testclient.TestClient(self.app) as test_client:
      response = test_client.request(
          "POST", "/execute_deep_research_agent", json=request_data
      )

      self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
  unittest.main()
