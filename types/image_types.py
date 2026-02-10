"""Image generation types."""

import enum


class AspectRatio(enum.Enum):
  """Aspect ratio of generated images."""

  RATIO_16_9 = '16:9'
  RATIO_1_1 = '1:1'
  RATIO_3_4 = '3:4'
  RATIO_4_3 = '4:3'
  RATIO_9_16 = '9:16'


class ImageSafetyLevel(enum.Enum):
  """Safety filtering level for generated images."""

  BLOCK_FEW = 'block_few'
  BLOCK_FEWEST = 'block_fewest'
  BLOCK_MOST = 'block_most'
  BLOCK_SOME = 'block_some'
