"""Tests for environment_util."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized


class EnvironmentUtilTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('dev', 'dev', 'dev'),
      ('prod', 'prod', 'prod'),
      ('staging', 'staging', 'staging'),
      ('autopush', 'autopush', 'autopush'),
      ('none', None, 'dev'),
  )
  @mock.patch.object(flags, 'get_opal_adk_environment', autospec=True)
  def test_get_opal_adk_environment(self, env_value, expected, mock_get_env):
    mock_get_env.return_value = env_value
    self.assertEqual(environment_util.get_opal_adk_environment(), expected)

  @parameterized.named_parameters(
      ('prod', 'prod', True),
      ('dev', 'dev', False),
      ('staging', 'staging', False),
  )
  @mock.patch.object(flags, 'get_opal_adk_environment', autospec=True)
  def test_is_prod_environment(self, env_value, expected, mock_get_env):
    mock_get_env.return_value = env_value
    self.assertEqual(environment_util.is_prod_environment(), expected)

  @parameterized.named_parameters(
      ('staging', 'staging', True),
      ('dev', 'dev', False),
      ('prod', 'prod', False),
  )
  @mock.patch.object(flags, 'get_opal_adk_environment', autospec=True)
  def test_is_staging_environment(self, env_value, expected, mock_get_env):
    mock_get_env.return_value = env_value
    self.assertEqual(environment_util.is_staging_environment(), expected)

  @parameterized.named_parameters(
      ('staging', 'staging', True),
      ('autopush', 'autopush', True),
      ('dev', 'dev', False),
      ('prod', 'prod', False),
  )
  @mock.patch.object(flags, 'get_opal_adk_environment', autospec=True)
  def test_is_autopush_or_staging(self, env_value, expected, mock_get_env):
    mock_get_env.return_value = env_value
    self.assertEqual(environment_util.is_autopush_or_staging(), expected)

  @parameterized.named_parameters(
      ('not_staging', 'dev', True),
      ('is_staging', 'staging', False),
  )
  @mock.patch.object(flags, 'get_opal_adk_environment', autospec=True)
  @mock.patch('logging.info', autospec=True)
  def test_log_if_not_staging(
      self, env_value, should_log, mock_logging, mock_get_env
  ):
    mock_get_env.return_value = env_value
    environment_util.log_if_not_staging('test message %s', 'arg')
    if should_log:
      mock_logging.assert_called_once_with('test message %s', 'arg')
    else:
      mock_logging.assert_not_called()


if __name__ == '__main__':
  absltest.main()
