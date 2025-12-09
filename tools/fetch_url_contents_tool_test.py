"""Unit tests for fetch_url_contents."""

import unittest
from unittest import mock
from opal_adk.tools import fetch_url_contents_tool
import requests


class FetchUrlContentsTest(unittest.TestCase):

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
    """Tests that fetch_url raises an exception on failure."""
    mock_get.side_effect = requests.exceptions.RequestException(
        'Failed to fetch'
    )

    url = 'http://example.com'
    with self.assertRaises(requests.exceptions.RequestException):
      fetch_url_contents_tool.fetch_url(url)

  def test_convert_to_markdown(self):
    """Tests that convert_to_markdown converts HTML to markdown."""
    html_content = '<html><body><h1>Title</h1><p>Hello world!</p></body></html>'
    expected_markdown = '# Title\n\nHello world!\n\n'
    result = fetch_url_contents_tool._convert_to_markdown(html_content)
    self.assertEqual(expected_markdown.strip(), result.strip())

  @mock.patch('requests.get')
  def test_fetch_url_raises_for_status(self, mock_get):
    """Tests that fetch_url raises an exception if raise_for_status does."""
    mock_response = mock.Mock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        '404 Not Found'
    )
    mock_get.return_value = mock_response

    url = 'http://example.com/404'
    with self.assertRaises(requests.exceptions.HTTPError):
      fetch_url_contents_tool.fetch_url(url)


if __name__ == '__main__':
  unittest.main()
