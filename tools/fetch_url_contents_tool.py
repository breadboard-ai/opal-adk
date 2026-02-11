"""Fetches the contents of a URL
"""

import logging
from typing import Optional
import html2text
import requests
from google.adk.tools import base_tool
from google.adk.tools import tool_context
from google.rpc import code_pb2
from opal_adk.error_handling import opal_adk_error


def _convert_to_markdown(site_contents: str) -> str:
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


class FetchUrlContentsTool(base_tool.BaseTool):
  """Fetches the contents of a webpage and returns the contents as markdown."""

  def __init__(self):
    super().__init__(
        name="OpalAdkFetchUrlContentsTool",
        description=(
            "Fetches the contents of a webpage and returns the contents as"
            " markdown."
        ),
    )

  def __call__(
      self, url: str, context: Optional[tool_context.ToolContext] = None
  ) -> str:
    """Fetches the contents of a URL.

    This method attempts to fetch the contents of the given URL. It uses the
    requests library.

    Args:
      url: The URL to fetch.
      context: The tool context.

    Returns:
      The contents of the URL as a string.

    Raises:
      requests.exceptions.RequestException: If the URL cannot be fetched.
    """
    try:
      response = requests.get(url)
      response.raise_for_status()
      return _convert_to_markdown(response.text)
    except requests.exceptions.RequestException as e:
      logging.info("fetch_url_contents_tool: failed to fetch url with exception: %s", e)
      raise opal_adk_error.OpalAdkError(
          logged=f"fetch_url_contents_tool: failed to fetch url with exception: {e}",
          status_message="Failed to fetch URL contents",
          status_code=code_pb2.UNAVAILABLE,
          details=str(e),
      ) from e


fetch_url = FetchUrlContentsTool()
