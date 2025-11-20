"""Constants related to Opal workflow steps."""

import enum


class OpalStepName(enum.Enum):
  """Names of the steps in the Opal workflow."""

  DEEP_RESEARCH = 'deep_research'
  MAP_SEARCH = 'map_search'
  WEB_SEARCH = 'web_search'
  GENERATE_WEBPAGE = 'generate_webpage'
