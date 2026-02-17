"""Module for generating images using Gemini models with consistency constraints."""

from typing import Any
from google.adk.tools import tool_context as tc
from google.genai import types
from opal_adk.error_handling import opal_adk_error
from opal_adk.tools.generate.generate_utils import gemini_generate_image
from opal_adk.types import image_types
from opal_adk.types import models
from google.rpc import code_pb2

ToolContext = tc.ToolContext

_INPUT_IMAGE_KEY = 'input_image'
_AI_IMAGE_TOOL_PREFIX = """Generate the image(s) below with consistent style (and characters as applicable).

🚨 CRITICAL: If the request asks for MULTIPLE images (e.g., "generate 3 images", "create 5 different scenes"), 
you MUST generate the EXACT number of SEPARATE images as individual outputs, NOT multiple images combined into 
a single image. Each image should be a distinct, standalone image file.

For example:
- If asked for "3 images of cats", generate 3 separate cat images
- If asked for "5 story scenes", generate 5 separate scene images
- Do NOT create a collage or grid combining multiple images into one

Pay careful attention to the exact number of images requested and ensure you generate that many separate images.

"""


async def generate_images(
    prompt: str,
    model: str,
    aspect_ratio: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
  """Generates one or more images based on a prompt and optionally.

  Args:
    prompt: This model can generate multiple images from a single prompt.
      Especially when looking for consistency across images (for instance, when
      generating video keyframes), this is a very useful capability. Be specific
      about how many images to generate. When composing the prompt, be as
      descriptive as possible. Describe the scene, don't just list keywords. The
      model's core strength is its deep language understanding. A narrative,
      descriptive paragraph will almost always produce a better, more coherent
      image than a list of disconnected words. This function allows you to use
      multiple input images to compose a new scene or transfer the style from
      one image to another. Here are some possible applications: -
      Text-to-Image: Generate high-quality images from simple or complex text
      descriptions. Provide a text prompt and no images as input. - Image +
      Text-to-Image (Editing): Provide an image and use the text prompt to add,
      remove, or modify elements, change the style, or adjust the color grading.
      - Multi-Image to Image (Composition & style transfer): Use multiple input
      images to compose a new scene or transfer the style from one image to
      another. - High-Fidelity text rendering: Accurately generate images that
      contain legible and well-placed text, ideal for logos, diagrams, and
      posters.
    model: The Gemini model to use for image generation. How to choose the right
      model -- choose "pro" to accurately generate images that contain legible
      and well-placed text, ideal for logos, diagrams, and posters. This model
      is designed for professional asset production and complex instructions -
      choose "flash" for speed and efficiency. This model is optimized for
      high-volume, low-latency tasks.
    aspect_ratio: The aspect ratio of the generated images. Supported values are
      ["1:1", "3:4", "4:3", "9:16", "16:9"].
    tool_context: ToolContext passed as part of the ADK tool execution. This
      will contain any images that were added by the user as a reference.

  Returns:
    A dict containing the status, message and dictionary of image metadata.
  """
  try:
    simple_model = models.SimpleModel(model)
  except ValueError:
    raise opal_adk_error.OpalAdkError(
        status_code=code_pb2.INVALID_ARGUMENT,
        status_message=(
            f'generate_imagesInvalid model name: {model}, expected one of'
            ' ["pro", "flash"]'
        ),
    )

  model_name = models.simple_model_to_image_model(simple_model).value

  parsed_aspect_ratio = image_types.AspectRatio(aspect_ratio)

  all_parts = [types.Part.from_text(text=_AI_IMAGE_TOOL_PREFIX + prompt)]
  input_images = await tool_context.load_artifact(_INPUT_IMAGE_KEY)
  if input_images:
    all_parts.append(input_images)

  generated_images = gemini_generate_image.gemini_generate_images(
      parts=all_parts,
      aspect_ratio=parsed_aspect_ratio,
      model_name=model_name,
  )

  metadata = {}
  for i, (image_bytes, mime_type) in enumerate(generated_images):
    image = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    image_name = 'output_image_' + str(i)
    tool_context.save_artifact(filename=image_name, artifact=image)
    metadata['image_name'] = image_name
    metadata['mime_type'] = mime_type
    tool_context.state['image_generated'] = (
        'Successfully generated and stored an image with image_name:'
        f' {image_name} and mime_type: {mime_type}.  The prompt used '
        f'for image generation was: {prompt}'
    )

  return {
      'status': 'success',
      'message': 'Successfully generated images',
      'metadata': metadata,
  }
