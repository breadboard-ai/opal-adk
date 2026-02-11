"""A tool for searching Google Maps using the Google Maps Platform APIs.

This tool provides a function to connect to the Google Maps Places API
and perform text-based searches for locations and establishments.
"""

import logging
from typing import Any

from google.adk.tools import base_tool
from google.adk.tools import tool_context
import googlemaps
from googlemaps import exceptions
from opal_adk import flags
from opal_adk.error_handling import opal_adk_error

from google.rpc import code_pb2


def _format_results(query: str, results: dict[str, Any] | str) -> str:
  """Formats Google Maps search results into a readable string.

  Args:
    query: The original search query.
    results: The search results, either as a pre-formatted string or a dict
      containing place details. Expected dict structure includes a 'places' key,
      where each place is a dict with keys like 'displayName', 'websiteUri',
      'editorialSummary', 'formattedAddress', 'rating', and 'userRatingCount'.

  Returns:
    A formatted string of the search results.
  """
  if isinstance(results, str):
    return f"""Search Query: {query}

## Google Places Search Results

{results}"""

  formatted_places = []
  for place in results.get("results", []):
    display_name = place.get("displayName", {}).get("text", "N/A")
    website_uri = place.get("websiteUri")
    title = f"[{display_name}]({website_uri})" if website_uri else display_name

    editorial_summary = place.get("editorialSummary", {}).get("text", "")
    formatted_address = place.get("formattedAddress", "N/A")
    rating = place.get("rating", "N/A")
    user_rating_count = place.get("userRatingCount", "N/A")

    formatted_place = f"""- {title}
  {editorial_summary}
  Address: {formatted_address}
  User Rating: {rating} ({user_rating_count} reviews)"""
    formatted_places.append(formatted_place)

  places_str = "\n\n".join(formatted_places)

  return f"""Search Query: {query}

## Google Places Search Results

{places_str}"""


class MapSearchTool(base_tool.BaseTool):
  """A tool for searching Google Maps Places API.

  This tool provides a callable interface to perform text-based searches
  for locations and establishments using the Google Maps Platform Places API.
  """

  def __init__(self):
    super().__init__(
        name="OpalAdkMapSearch",
        description="Performs a places search on Google Maps.",
    )

  def __call__(
      self, query: str, context: tool_context.ToolContext
  ) -> dict[str, Any]:
    """Connects to Google Maps Places API and performs a text search.

    This function uses the publicly available `googlemaps` Python client library
    to interact with the Google Maps Platform Places API.

    Args:
      query: The text string to search for (e.g., "pizza near me").
      context: The tool context.

    Returns:
      A dictionary containing the search results from the Places API.
      Returns an empty dictionary if an error occurs.
    """
    api_key = flags.get_maps_api_key()
    if not api_key:
      raise opal_adk_error.OpalAdkError(
          logged="map_search_tool: Google Maps Platform API key is required.",
          status_message=(
              "map_search_tool: Google Maps Platform API key is required."
          ),
          status_code=code_pb2.FAILED_PRECONDITION,
      )

    gmaps = googlemaps.Client(key=api_key)

    try:
      results = gmaps.places(query=query)
      return {"result": _format_results(query, results)}
    except exceptions.ApiError as e:
      logging.exception("Google Maps API Error: %s", e)
      raise opal_adk_error.OpalAdkError(
          status_message="map_search_tool: Error with Google Maps API key.",
          status_code=code_pb2.FAILED_PRECONDITION,
          internal_details=f"map_search_tool: error details: {e}",
      ) from e
    except Exception as e:
      logging.exception("An unexpected error occurred: %s", e)
      raise opal_adk_error.OpalAdkError(
          status_message="maps_search_tool: Internal error.",
          status_code=code_pb2.INTERNAL,
          internal_details=f"map_search_tool: Intenal error details: {e}",
      ) from e
