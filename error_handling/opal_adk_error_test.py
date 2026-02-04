"""Tests for opal_adk_error."""

import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from google.genai import errors as genai_errors
from opal_adk.error_handling import opal_adk_error
from google.rpc import code_pb2


class OpalAdkErrorTest(parameterized.TestCase):

  def test_opal_adk_error_init_defaults(self, mock_is_prod):
    error = opal_adk_error.OpalAdkError()
    self.assertEqual(error.status_message, opal_adk_error.GENERIC_ERROR_MESSAGE)
    self.assertEqual(error.error_code, code_pb2.UNKNOWN)
    self.assertEqual(error.details, '')
    self.assertEqual(str(error), opal_adk_error.GENERIC_ERROR_MESSAGE)

  def test_opal_adk_error_init_custom(self, mock_is_prod):
    error = opal_adk_error.OpalAdkError(
        logged='logged_msg',
        status_message='status_msg',
        status_code=code_pb2.INVALID_ARGUMENT,
        details='details_msg',
        rewritten_intent='rewritten',
    )
    self.assertEqual(error.status_message, 'status_msg')
    self.assertEqual(error.error_code, code_pb2.INVALID_ARGUMENT)
    self.assertEqual(error.details, 'details_msg')
    self.assertEqual(error.rewritten_intent, 'rewritten')
    self.assertEqual(str(error), 'logged_msg')

  def test_opal_adk_error_external_message(self):
    error = opal_adk_error.OpalAdkError(
        status_message='msg',
        status_code=code_pb2.NOT_FOUND,
        details='det',
    )
    external_json = error.external_message()
    external_dict = json.loads(external_json)
    self.assertEqual(
        external_dict,
        {
            'code': 'NOT_FOUND',
            'message': 'msg',
            'details': 'det',
        },
    )

  def test_chat_error_defaults(self):
    base_error = ValueError('base')
    error = opal_adk_error.ChatError(base_error)
    self.assertIn('Chat error', str(error))
    self.assertIn(opal_adk_error.GENERIC_CHAT_BASED_PREFIX, error.details)

  def test_chat_error_full_message(self):
    base_error = ValueError('base')
    error = opal_adk_error.ChatError(base_error, full_chat_message='full_msg')
    self.assertEqual(error.details, 'full_msg')
    self.assertIn('full_msg', str(error))

  def test_chat_error_with_opal_error(self):
    base_error = opal_adk_error.OpalAdkError(details='base_details')
    error = opal_adk_error.ChatError(base_error, chat_prefix='prefix: ')
    self.assertIn('prefix: Details: base_details', error.details)

  def test_get_opal_adk_error_idempotent(self):
    error = opal_adk_error.OpalAdkError(status_message='original')
    result = opal_adk_error.get_opal_adk_error(error)
    self.assertIs(result, error)

  @parameterized.named_parameters(
      (
          'resource_exhausted',
          genai_errors.ClientError(
              code=code_pb2.RESOURCE_EXHAUSTED,
              response_json={'message': 'exhausted'},
          ),
          code_pb2.RESOURCE_EXHAUSTED,
          'higher load',
      ),
      (
          'other_client_error',
          genai_errors.ClientError(
              code=code_pb2.INVALID_ARGUMENT,
              response_json={'message': 'other'},
          ),
          code_pb2.INTERNAL,
          opal_adk_error.GENERIC_MODEL_ERROR_MESSAGE,
      ),
      (
          'server_error',
          genai_errors.ServerError(
              code=500,
              response_json={'message': 'server'},
          ),
          code_pb2.INTERNAL,
          opal_adk_error.GENERIC_MODEL_ERROR_MESSAGE,
      ),
      (
          'unknown_error',
          ValueError('unknown'),
          code_pb2.UNKNOWN,
          opal_adk_error.GENERIC_ERROR_MESSAGE,
      ),
  )
  def test_get_opal_adk_error(
      self, exception, expected_code, expected_message_part
  ):
    result = opal_adk_error.get_opal_adk_error(exception)
    self.assertEqual(result.error_code, expected_code)
    self.assertIn(expected_message_part, result.status_message)

  @parameterized.named_parameters(
      (
          'standard',
          True,
          'status\ndetails',
      ),
  )
  @mock.patch.object(environment_util, 'is_prod_environment', autospec=True)
  def test_get_error_as_chat_message(self, is_prod, expected_msg, mock_is_prod):
    mock_is_prod.return_value = is_prod
    error = opal_adk_error.OpalAdkError(
        logged='internal',  # Provide logged to ensure internal_details is set
        status_message='status',
        details='details',
    )
    msg = opal_adk_error.get_error_as_chat_message(error)
    self.assertEqual(msg, expected_msg)


if __name__ == '__main__':
  absltest.main()
