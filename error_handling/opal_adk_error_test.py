"""Tests for opal_adk_error."""

import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from google.genai import errors as genai_errors
from opal_adk.error_handling import opal_adk_error
from google.rpc import code_pb2

# copybara:strip_begin
from google3.learning.language.tunelab.contrib.demos.intent2app.infra import environment_util
# copybara:strip_end


class OpalAdkErrorTest(parameterized.TestCase):

  # copybara:strip_begin
  @mock.patch.object(environment_util, 'is_prod_environment', autospec=True)
  # copybara:strip_end
  def test_opal_adk_error_init_defaults(self, mock_is_prod):
    # copybara:strip_begin
    mock_is_prod.return_value = True
    # copybara:strip_end
    error = opal_adk_error.OpalAdkError()
    self.assertEqual(error.status_message, opal_adk_error.GENERIC_ERROR_MESSAGE)
    self.assertEqual(error.error_code, code_pb2.UNKNOWN)
    self.assertEqual(error.details, '')
    # copybara:strip_begin
    self.assertEqual(error.internal_details, '')
    # copybara:strip_end
    self.assertEqual(str(error), opal_adk_error.GENERIC_ERROR_MESSAGE)

  # copybara:strip_begin
  @mock.patch.object(environment_util, 'is_prod_environment', autospec=True)
  # copybara:strip_end
  def test_opal_adk_error_init_custom(self, mock_is_prod):
    # copybara:strip_begin
    # In non-prod, internal_details is overwritten by logged
    mock_is_prod.return_value = False
    # copybara:strip_end
    error = opal_adk_error.OpalAdkError(
        logged='logged_msg',
        status_message='status_msg',
        status_code=code_pb2.INVALID_ARGUMENT,
        details='details_msg',
        # copybara:strip_begin
        internal_details='internal_msg',
        # copybara:strip_end
        rewritten_intent='rewritten',
    )
    self.assertEqual(error.status_message, 'status_msg')
    self.assertEqual(error.error_code, code_pb2.INVALID_ARGUMENT)
    self.assertEqual(error.details, 'details_msg')
    # copybara:strip_begin
    # Expect logged_msg in non-prod because implementation overwrites it
    self.assertEqual(error.internal_details, 'logged_msg')
    # copybara:strip_end
    self.assertEqual(error.rewritten_intent, 'rewritten')
    self.assertEqual(str(error), 'logged_msg')

  # copybara:strip_begin
  @mock.patch.object(environment_util, 'is_prod_environment', autospec=True)
  def test_opal_adk_error_init_non_prod_internal_details(self, mock_is_prod):
    mock_is_prod.return_value = False
    error = opal_adk_error.OpalAdkError(logged='logged')
    self.assertEqual(error.internal_details, 'logged')

  # copybara:strip_end

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

  # copybara:strip_begin
  @parameterized.named_parameters(
      ('prod', True, opal_adk_error.GENERIC_ERROR_MESSAGE, code_pb2.UNKNOWN),
      ('non_prod', False, 'status', code_pb2.INTERNAL),
  )
  @mock.patch.object(environment_util, 'is_prod_environment', autospec=True)
  def test_internal_details_error(
      self, is_prod, expected_status, expected_code, mock_is_prod
  ):
    mock_is_prod.return_value = is_prod
    error = opal_adk_error.InternalDetailsError(
        logged='logged', status_message='status'
    )
    self.assertEqual(error.status_message, expected_status)
    self.assertEqual(error.error_code, expected_code)
    self.assertEqual(error.internal_details, 'logged')
    if not is_prod:
      self.assertEqual(str(error), 'logged')

  # copybara:strip_end

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

  # copybara:strip_begin
  @mock.patch.object(environment_util, 'is_prod_environment', autospec=True)
  def test_chat_error_non_prod_internal_details(self, mock_is_prod):
    mock_is_prod.return_value = False
    # Provide logged so internal_details gets set to it in non-prod
    base_error = opal_adk_error.OpalAdkError(
        logged='internal', internal_details='internal'
    )
    error = opal_adk_error.ChatError(base_error)
    # ChatError constructs logged_message which overwrites internal_details
    self.assertIn('Chat error', error.internal_details)
    self.assertIn('internal', str(error))

  # copybara:strip_end

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
      # copybara:strip_begin
      (
          'non_prod',
          False,
          'status\ninternal',
      ),
      # copybara:strip_end
  )
  @mock.patch.object(environment_util, 'is_prod_environment', autospec=True)
  def test_get_error_as_chat_message(self, is_prod, expected_msg, mock_is_prod):
    mock_is_prod.return_value = is_prod
    error = opal_adk_error.OpalAdkError(
        logged='internal',  # Provide logged to ensure internal_details is set
        status_message='status',
        details='details',
        # copybara:strip_begin
        internal_details='internal',
        # copybara:strip_end
    )
    msg = opal_adk_error.get_error_as_chat_message(error)
    self.assertEqual(msg, expected_msg)


if __name__ == '__main__':
  absltest.main()
