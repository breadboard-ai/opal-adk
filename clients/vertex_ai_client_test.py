"""Tests for vertex_ai_client."""

import unittest
from unittest import mock

from absl.testing import absltest
from google import genai
from opal_adk.clients import vertex_ai_client
from opal_adk.error_handling import opal_adk_error

from google.rpc import code_pb2


class VertexAiClientTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_client = mock.patch.object(genai, 'Client', autospec=True).start()
    self.addCleanup(mock.patch.stopall)

  def test_create_vertex_ai_client_success(self):
    client = vertex_ai_client.create_vertex_ai_client()
    self.assertEqual(client, self.mock_client.return_value)

  def test_create_vertex_ai_client_failure(self):
    self.mock_client.side_effect = RuntimeError('Initialization failed')
    with self.assertRaises(opal_adk_error.OpalAdkError) as cm:
      vertex_ai_client.create_vertex_ai_client()

    self.assertEqual(cm.exception.error_code, code_pb2.INTERNAL)
    self.assertIn(
        'Could not initialize genai.Client', cm.exception.status_message
    )


if __name__ == '__main__':
  absltest.main()
