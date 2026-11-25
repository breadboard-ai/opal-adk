"""Unit tests for fetch_url_contents."""

import json
from unittest import mock
from absl.testing import absltest
from opal_adk.error_handling import opal_adk_error
from opal_adk.tools import fetch_url_contents_tool
import requests


class FetchUrlContentsTest(absltest.TestCase):

  @mock.patch('requests.get')
  def test_fetch_url_success(self, mock_get):
    """Tests that fetch_url returns markdown content on success."""
    mock_response = mock.Mock()
    mock_response.text = (
        '<html><body><h1>Test</h1><p>Some text</p>'
        '<a href="http://link.com">link</a>'
        '<img src="http://image.com/img.png" alt="alt text"></body></html>'
    )
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    url = 'http://example.com'
    result = fetch_url_contents_tool.fetch_url(url)

    mock_get.assert_called_once_with(url)
    mock_response.raise_for_status.assert_called_once()
    self.assertIn('# Test', result)
    self.assertIn('Some text', result)
    self.assertIn('[link](http://link.com)', result)
    self.assertIn('![alt text](http://image.com/img.png)', result)

  @mock.patch('requests.get')
  def test_fetch_url_failure(self, mock_get):
    """Tests that fetch_url hides raw network errors and logs them."""
    raw_error = (
        "HTTPConnectionPool(host='10.0.0.5', port=8500): Max retries exceeded"
        ' with url: /secret (Caused by NewConnectionError: [Errno 111]'
        ' Connection refused)'
    )
    cause = requests.exceptions.ConnectionError(raw_error)
    mock_get.side_effect = cause

    url = 'http://10.0.0.5:8500/secret'
    with self.assertLogs(level='WARNING') as log_cm:
      with self.assertRaises(opal_adk_error.OpalAdkError) as cm:
        fetch_url_contents_tool.fetch_url(url)

    err = cm.exception
    expected_details = 'The requested URL could not be fetched.'
    self.assertEqual(err.details, expected_details)
    self.assertEqual(
        json.loads(err.external_message()),
        {
            'code': 'UNAVAILABLE',
            'message': 'Failed to fetch URL contents',
            'details': expected_details,
        },
    )
    chat_details = opal_adk_error.ChatError(err).details
    self.assertIn(expected_details, chat_details)
    for leaked_fragment in (
        '10.0.0.5',
        '8500',
        '/secret',
        'Connection refused',
        raw_error,
    ):
      self.assertNotIn(leaked_fragment, err.details)
      self.assertNotIn(leaked_fragment, err.external_message())
      self.assertNotIn(leaked_fragment, chat_details)

    self.assertIn(raw_error, str(err))
    self.assertTrue(any(raw_error in msg for msg in log_cm.output))
    self.assertIs(err.__cause__, cause)

  def test_convert_to_markdown(self):
    """Tests that convert_to_markdown converts HTML to markdown."""
    html_content = '<html><body><h1>Title</h1><p>Hello world!</p></body></html>'
    expected_markdown = '# Title\n\nHello world!\n\n'
    result = fetch_url_contents_tool._convert_to_markdown(html_content)
    self.assertEqual(expected_markdown.strip(), result.strip())

  @mock.patch('requests.get')
  def test_fetch_url_raises_for_status(self, mock_get):
    """Tests that HTTP status errors are sanitized externally and logged."""
    raw_error = (
        '404 Client Error: Not Found for url:'
        ' http://169.254.169.254/internal/admin'
    )
    mock_response = mock.Mock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        raw_error
    )
    mock_get.return_value = mock_response

    url = 'http://169.254.169.254/internal/admin'
    with self.assertLogs(level='WARNING') as log_cm:
      with self.assertRaises(opal_adk_error.OpalAdkError) as cm:
        fetch_url_contents_tool.fetch_url(url)

    err = cm.exception
    expected_details = 'The requested URL could not be fetched.'
    self.assertEqual(err.details, expected_details)
    chat_details = opal_adk_error.ChatError(err).details
    for leaked_fragment in (
        '404',
        'Not Found',
        '169.254.169.254',
        '/internal/admin',
    ):
      self.assertNotIn(leaked_fragment, err.details)
      self.assertNotIn(leaked_fragment, err.external_message())
      self.assertNotIn(leaked_fragment, chat_details)

    self.assertIn(raw_error, str(err))
    self.assertTrue(any(raw_error in msg for msg in log_cm.output))

  @mock.patch('requests.get')
  def test_fetch_url_timeout_hides_internal_hostname(self, mock_get):
    """Tests that timeouts hide internal hosts and match other fetch errors."""
    timeout_error = (
        "HTTPConnectionPool(host='db.internal.example', port=5432):"
        ' Read timed out. (read timeout=10)'
    )
    mock_get.side_effect = requests.exceptions.ReadTimeout(timeout_error)

    with self.assertLogs(level='WARNING') as log_cm:
      with self.assertRaises(opal_adk_error.OpalAdkError) as timeout_cm:
        fetch_url_contents_tool.fetch_url('http://db.internal.example:5432/')

    err = timeout_cm.exception
    expected_details = 'The requested URL could not be fetched.'
    self.assertEqual(err.details, expected_details)
    chat_details = opal_adk_error.ChatError(err).details
    for leaked_fragment in ('db.internal.example', '5432', 'timed out'):
      self.assertNotIn(leaked_fragment, err.details)
      self.assertNotIn(leaked_fragment, err.external_message())
      self.assertNotIn(leaked_fragment, chat_details)

    self.assertIn(timeout_error, str(err))
    self.assertTrue(any(timeout_error in msg for msg in log_cm.output))

    # Different network failure classes must return identical external payloads
    # so callers cannot distinguish closed, filtered, or HTTP error responses.
    mock_get.side_effect = requests.exceptions.ConnectionError(
        'Connection refused'
    )
    with self.assertLogs(level='WARNING'):
      with self.assertRaises(opal_adk_error.OpalAdkError) as conn_cm:
        fetch_url_contents_tool.fetch_url('http://127.0.0.1:1/')
    self.assertEqual(
        err.external_message(), conn_cm.exception.external_message()
    )


if __name__ == '__main__':
  absltest.main()
