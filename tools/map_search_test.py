"""Unit tests for map_search."""

import logging
from unittest import mock

from absl import flags as absl_flags
from googlemaps import exceptions
from opal_adk import flags
from opal_adk.tools import map_search

from google3.testing.pybase import googletest

FLAGS = absl_flags.FLAGS


class MapSearchTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_client_instance = mock.MagicMock()
    self.mock_client = self.enter_context(
        mock.patch("googlemaps.Client", return_value=self.mock_client_instance)
    )
    self.mock_flags = self.enter_context(
        mock.patch.object(
            flags, "get_maps_api_key", return_value="fake_api_key"
        )
    )
    self.mock_logging = self.enter_context(mock.patch.object(logging, "error"))

  def test_format_results_string_input(self):
    query = "test query"
    results = "string results"
    formatted = map_search._format_results(query, results)
    self.assertIn(query, formatted)
    self.assertIn(results, formatted)
    self.assertStartsWith(
        formatted, "Search Query: test query\n## Google Places Search Results\n"
    )

  def test_format_results_dict_input(self):
    query = "test query"
    results = {
        "places": [{
            "displayName": {"text": "Place 1"},
            "websiteUri": "http://place1.com",
            "editorialSummary": {"text": "Summary 1"},
            "formattedAddress": "Address 1",
            "rating": 4.5,
            "userRatingCount": 100,
        }]
    }
    formatted = map_search._format_results(query, results)
    self.assertIn("[Place 1](http://place1.com)", formatted)
    self.assertIn("Summary 1", formatted)
    self.assertIn("Address: Address 1", formatted)
    self.assertIn("User Rating: 4.5 (100 reviews)", formatted)

  def test_format_results_dict_no_website(self):
    query = "test query"
    results = {
        "places": [{
            "displayName": {"text": "Place 1"},
            "editorialSummary": {"text": "Summary 1"},
            "formattedAddress": "Address 1",
            "rating": 4.5,
            "userRatingCount": 100,
        }]
    }
    formatted = map_search._format_results(query, results)
    self.assertIn("- Place 1", formatted)
    self.assertNotIn("http://place1.com", formatted)
    self.assertNotIn("[Place 1]", formatted)

  def test_format_results_missing_fields(self):
    query = "test query"
    results = {
        "places": [{
            "displayName": {"text": "Place 1"},
        }]
    }
    formatted = map_search._format_results(query, results)
    self.assertIn("- Place 1", formatted)
    self.assertIn("Address: N/A", formatted)
    self.assertIn("User Rating: N/A (N/A reviews)", formatted)

  def test_format_results_empty_places(self):
    query = "test query"
    results = {"places": []}
    formatted = map_search._format_results(query, results)
    self.assertEqual(
        formatted,
        "Search Query: test query\n\n## Google Places Search Results\n\n",
    )

  def test_search_google_maps_places_no_api_key(self):
    with self.assertRaisesRegex(
        ValueError, "Google Maps Platform API key is required."
    ):
      map_search.search_google_maps_places(query="test", api_key="")

  def test_search_google_maps_places_success(self):
    self.mock_client_instance.places.return_value = "results"
    res = map_search.search_google_maps_places(
        query="test query", api_key="provided_key"
    )
    self.mock_client.assert_called_once_with(key="fake_api_key")
    self.mock_client_instance.places.assert_called_once_with(query="test query")
    self.assertEqual(
        res,
        "Search Query: test query\n## Google Places Search Results\nresults",
    )

  def test_search_google_maps_places_api_error(self):
    self.mock_client_instance.places.side_effect = exceptions.ApiError("error")
    res = map_search.search_google_maps_places(
        query="test", api_key="provided_key"
    )
    self.assertEqual(res, {})
    self.mock_logging.assert_called_once()

  def test_search_google_maps_places_exception(self):
    self.mock_client_instance.places.side_effect = Exception("error")
    res = map_search.search_google_maps_places(
        query="test", api_key="provided_key"
    )
    self.assertEqual(res, {})
    self.mock_logging.assert_called_once()


if __name__ == "__main__":
  FLAGS.set_default("opal_adk_gcp_service_account", "dummy_sa")
  FLAGS.set_default("opal_adk_gcp_location", "dummy_location")
  FLAGS.set_default("opal_adk_gcp_project_id", "dummy_project")
  FLAGS.set_default("opal_adk_maps_api_key", "dummy_key")
  googletest.main()
