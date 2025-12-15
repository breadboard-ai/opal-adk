"""Flags used across the Opal ADK."""

from typing import NewType
from absl import flags

ServiceAccount = NewType("ServiceAccount", str)
ProjectId = NewType("ProjectId", str)
Location = NewType("Location", str)
MapsAPIKey = NewType("MapsAPIKey", str)


_OPAL_ADK_GCP_SERVICE_ACCOUNT = flags.DEFINE_string(
    "opal_adk_gcp_service_account",
    required=True,
    default=None,
    help="GCP service account to use for Opal ADK.",
)
_OPAL_ADK_GCP_LOCATION = flags.DEFINE_string(
    "opal_adk_gcp_location",
    required=True,
    default=None,
    help="GCP location for Opal ADK resources.",
)
_OPAL_ADK_GCP_PROJECT_ID = flags.DEFINE_string(
    "opal_adk_gcp_project_id",
    required=True,
    default=None,
    help="GCP project ID for Opal ADK.",
)

_OPAL_ADK_MAPS_API_KEY = flags.DEFINE_string(
    "opal_adk_maps_api_key",
    required=False,
    default=None,
    help="API key for using the Google Maps API.",
)


def get_service_account() -> ServiceAccount:
  try:
    return ServiceAccount(_OPAL_ADK_GCP_SERVICE_ACCOUNT.value)
  except flags.UnparsedFlagAccessError:
    return ServiceAccount(_OPAL_ADK_GCP_SERVICE_ACCOUNT.default)


def get_location() -> Location:
  try:
    return Location(_OPAL_ADK_GCP_LOCATION.value)
  except flags.UnparsedFlagAccessError:
    return Location(_OPAL_ADK_GCP_LOCATION.default)


def get_project_id() -> ProjectId:
  try:
    return ProjectId(_OPAL_ADK_GCP_PROJECT_ID.value)
  except flags.UnparsedFlagAccessError:
    return ProjectId(_OPAL_ADK_GCP_PROJECT_ID.default)


def get_maps_api_key() -> MapsAPIKey:
  try:
    return MapsAPIKey(_OPAL_ADK_MAPS_API_KEY.value)
  except flags.UnparsedFlagAccessError:
    return MapsAPIKey(_OPAL_ADK_MAPS_API_KEY.default)
