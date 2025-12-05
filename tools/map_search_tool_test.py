"""Unit tests for map_search_tool."""

import logging
from unittest import mock

from absl import flags as absl_flags
from googlemaps import exceptions
from opal_adk import flags
from opal_adk.tools import map_search_tool

from google3.testing.pybase import googletest

FLAGS = absl_flags.FLAGS


class MapSearchToolTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_client_instance = mock.MagicMock()
    self.mock_client = self.enter_context(
        mock.patch(
            "googlemaps.Client",
            return_value=self.mock_client_instance,
            autospec=True,
        )
    )
    self.mock_flags = self.enter_context(
        mock.patch.object(
            flags,
            "get_maps_api_key",
            return_value="fake_api_key",
            autospec=True,
        )
    )
    self.mock_logging = self.enter_context(
        mock.patch.object(logging, "error", autospec=True)
    )
    self.tool = map_search_tool.MapSearchTool()

  def test_format_results_string_input(self):
    query = "test query"
    results = "string results"
    formatted = map_search_tool._format_results(query, results)
    self.assertIn(query, formatted)
    self.assertIn(results, formatted)
    self.assertStartsWith(
        formatted,
        "Search Query: test query\n\n## Google Places Search Results\n",
    )

  def test_format_results_dict_input(self):
    query = "test query"
    results = {
        "results": [{
            "displayName": {"text": "Place 1"},
            "websiteUri": "http://place1.com",
            "editorialSummary": {"text": "Summary 1"},
            "formattedAddress": "Address 1",
            "rating": 4.5,
            "userRatingCount": 100,
        }]
    }
    formatted = map_search_tool._format_results(query, results)
    self.assertIn("[Place 1](http://place1.com)", formatted)
    self.assertIn("Summary 1", formatted)
    self.assertIn("Address: Address 1", formatted)
    self.assertIn("User Rating: 4.5 (100 reviews)", formatted)

  def test_format_results_dict_no_website(self):
    query = "test query"
    results = {
        "results": [{
            "displayName": {"text": "Place 1"},
            "editorialSummary": {"text": "Summary 1"},
            "formattedAddress": "Address 1",
            "rating": 4.5,
            "userRatingCount": 100,
        }]
    }
    formatted = map_search_tool._format_results(query, results)
    self.assertIn("- Place 1", formatted)
    self.assertNotIn("http://place1.com", formatted)
    self.assertNotIn("[Place 1]", formatted)

  def test_format_results_missing_fields(self):
    query = "test query"
    results = {
        "results": [{
            "displayName": {"text": "Place 1"},
        }]
    }
    formatted = map_search_tool._format_results(query, results)
    self.assertIn("- Place 1", formatted)
    self.assertIn("Address: N/A", formatted)
    self.assertIn("User Rating: N/A (N/A reviews)", formatted)

  def test_format_results_empty_places(self):
    query = "test query"
    results = {"results": []}
    formatted = map_search_tool._format_results(query, results)
    self.assertEqual(
        formatted,
        "Search Query: test query\n\n## Google Places Search Results\n\n",
    )

  def test_call_no_api_key(self):
    self.mock_flags.return_value = None
    with self.assertRaisesRegex(
        ValueError, "Google Maps Platform API key is required."
    ):
      self.tool(query="test", context=mock.MagicMock())

  def test_call_success(self):
    self.mock_client_instance.places.return_value = "results"
    res = self.tool(query="test query", context=mock.MagicMock())
    self.mock_client.assert_called_once_with(key="fake_api_key")
    self.mock_client_instance.places.assert_called_once_with(query="test query")
    self.assertEqual(
        res,
        {
            "result": (
                "Search Query: test query\n\n## Google Places Search"
                " Results\n\nresults"
            )
        },
    )

  def test_call_api_error(self):
    self.mock_client_instance.places.side_effect = exceptions.ApiError("error")
    res = self.tool(query="test", context=mock.MagicMock())
    self.assertEqual(res, {})
    self.mock_logging.assert_called_once()

  def test_call_exception(self):
    self.mock_client_instance.places.side_effect = Exception("error")
    res = self.tool(query="test", context=mock.MagicMock())
    self.assertEqual(res, {})
    self.mock_logging.assert_called_once()


if __name__ == "__main__":
  FLAGS.set_default("opal_adk_gcp_service_account", "dummy_sa")
  FLAGS.set_default("opal_adk_gcp_location", "dummy_location")
  FLAGS.set_default("opal_adk_gcp_project_id", "dummy_project")
  FLAGS.set_default("opal_adk_maps_api_key", "dummy_key")
  googletest.main()
