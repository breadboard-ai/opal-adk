"""Flags used across the Opal ADK."""

from typing import NewType
from absl import flags

ServiceAccount = NewType("ServiceAccount", str)
ProjectId = NewType("ProjectId", str)
Location = NewType("Location", str)

_OPAL_ADK_GCP_SERVICE_ACCOUNT = flags.DEFINE_string(
    "opal_adk_gcp_service_account",
    required=True,
    default=None,
    help="GCP service account to use for Opal ADK.",
)
_OPAL_ADK_GCP_LOCATION = flags.DEFINE_string(
    "opal_adk_gcp_location",
    required=True,
    default="us-central1",
    help="GCP location for Opal ADK resources.",
)
_OPAL_ADK_GCP_PROJECT_ID = flags.DEFINE_string(
    "opal_adk_gcp_project_id",
    required=True,
    default=None,
    help="GCP project ID for Opal ADK.",
)


def get_service_account() -> ServiceAccount:
  return _OPAL_ADK_GCP_SERVICE_ACCOUNT.value


def get_location() -> Location:
  return _OPAL_ADK_GCP_LOCATION.value


def get_project_id() -> ProjectId:
  return _OPAL_ADK_GCP_PROJECT_ID.value
