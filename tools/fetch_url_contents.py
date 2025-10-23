"""Fetches the contents of a URL
"""

import logging
import html2text
import requests


def convert_to_markdown(site_contents: str) -> str:
  """Fetches the contents of a URL and returns them as markdown.

  This method fetches the HTML content of the given URL using `fetch_url`
  and then converts the HTML to markdown using the `html2text` library.

  Args:
    site_contents: The URL to fetch.

  Returns:
    The contents of the URL converted to markdown.

  Raises:
    requests.exceptions.RequestException: If the URL cannot be fetched.
  """
  h = html2text.HTML2Text()
  # Configure html2text to keep links and other formatting.
  h.ignore_links = False
  h.ignore_images = False
  h.body_width = 0  # Disable line wrapping
  markdown_content = h.handle(site_contents)
  return markdown_content


def fetch_url(url: str) -> str:
  """Fetches the contents of a URL.

  This method attempts to fetch the contents of the given URL. It uses the
  requests library.

  Args:
    url: The URL to fetch.

  Returns:
    The contents of the URL as a string.

  Raises:
    requests.exceptions.RequestException: If the URL cannot be fetched.
  """
  try:
    response = requests.get(url)
    response.raise_for_status()
    return convert_to_markdown(response.text)
  except requests.exceptions.RequestException as e:
    logging.info('fetch_url: failed to fetch url with exception: %s', e)
    raise e
