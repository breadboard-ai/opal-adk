"""Utility functions for environment setting and checks."""

import logging

from opal_adk import flags


def get_opal_adk_environment() -> str:
  """Returns the Opal ADK environment."""
  return flags.get_opal_adk_environment() or 'dev'


def is_prod_environment() -> bool:
  """Returns True if the current environment is prod."""
  return flags.get_opal_adk_environment() == 'prod'


def is_staging_environment() -> bool:
  """Returns True if the current environment is staging."""
  return flags.get_opal_adk_environment() == 'staging'


def is_autopush_or_staging() -> bool:
  """Returns True if the current environment is staging."""
  return flags.get_opal_adk_environment() in ('staging', 'autopush')


def log_if_not_staging(message: str, *args) -> None:
  """Logs the given message only if the current environment is not staging."""
  if not is_staging_environment():
    logging.info(message, *args)
