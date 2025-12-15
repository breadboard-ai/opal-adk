"""Tests for flags.py."""

import sys
from unittest import mock
from absl import flags
from absl.testing import absltest
from opal_adk import flags as opal_flags


class FlagsTest(absltest.TestCase):

  def test_get_service_account_success(self):
    with mock.patch.object(
        opal_flags, "_OPAL_ADK_GCP_SERVICE_ACCOUNT"
    ) as mock_flag:
      mock_flag.value = "test_service_account"
      self.assertEqual(opal_flags.get_service_account(), "test_service_account")

  def test_get_service_account_unparsed(self):
    with mock.patch.object(
        opal_flags, "_OPAL_ADK_GCP_SERVICE_ACCOUNT"
    ) as mock_flag:
      type(mock_flag).value = mock.PropertyMock(
          side_effect=flags.UnparsedFlagAccessError("msg")
      )
      mock_flag.default = None
      self.assertIsNone(opal_flags.get_service_account())

  def test_get_location_success(self):
    with mock.patch.object(opal_flags, "_OPAL_ADK_GCP_LOCATION") as mock_flag:
      mock_flag.value = "test_location"
      self.assertEqual(opal_flags.get_location(), "test_location")

  def test_get_location_unparsed(self):
    with mock.patch.object(opal_flags, "_OPAL_ADK_GCP_LOCATION") as mock_flag:
      type(mock_flag).value = mock.PropertyMock(
          side_effect=flags.UnparsedFlagAccessError("msg")
      )
      mock_flag.default = None
      self.assertIsNone(opal_flags.get_location())

  def test_get_project_id_success(self):
    with mock.patch.object(opal_flags, "_OPAL_ADK_GCP_PROJECT_ID") as mock_flag:
      mock_flag.value = "test_project_id"
      self.assertEqual(opal_flags.get_project_id(), "test_project_id")

  def test_get_project_id_unparsed(self):
    with mock.patch.object(opal_flags, "_OPAL_ADK_GCP_PROJECT_ID") as mock_flag:
      type(mock_flag).value = mock.PropertyMock(
          side_effect=flags.UnparsedFlagAccessError("msg")
      )
      mock_flag.default = None
      self.assertIsNone(opal_flags.get_project_id())

  def test_get_maps_api_key_success(self):
    with mock.patch.object(opal_flags, "_OPAL_ADK_MAPS_API_KEY") as mock_flag:
      mock_flag.value = "test_api_key"
      self.assertEqual(opal_flags.get_maps_api_key(), "test_api_key")

  def test_get_maps_api_key_unparsed(self):
    with mock.patch.object(opal_flags, "_OPAL_ADK_MAPS_API_KEY") as mock_flag:
      type(mock_flag).value = mock.PropertyMock(
          side_effect=flags.UnparsedFlagAccessError("msg")
      )
      mock_flag.default = None
      self.assertIsNone(opal_flags.get_maps_api_key())


if __name__ == "__main__":
  # Provide dummy values for required flags to satisfy absl parsing.
  sys.argv.extend([
      "--opal_adk_gcp_service_account=dummy",
      "--opal_adk_gcp_location=dummy",
      "--opal_adk_gcp_project_id=dummy",
  ])
  absltest.main()
