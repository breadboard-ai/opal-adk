"""Module for generating images using Gemini models with consistency constraints."""

from google.genai import types
from opal_adk.tools.generate.generate_utils import gemini_generate_image
from opal_adk.types import image_types
from opal_adk.types import models

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


def generate_images(
    prompt: str, model: str, images: list[types.Content], aspect_ratio: str
) -> list[types.Content]:
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
    images: A list of input images, specified as genai.types.Content.
    aspect_ratio: The aspect ratio of the generated images. Supported values are
      ["1:1", "3:4", "4:3", "9:16", "16:9"].

  Returns:
    A list of generated images.
  """
  simple_model = models.SimpleModel(model)
  model_name = models.simple_model_to_image_model(simple_model).value

  parsed_aspect_ratio = image_types.AspectRatio(aspect_ratio)

  all_parts = [types.Part.from_text(text=_AI_IMAGE_TOOL_PREFIX + prompt)]
  for content in images:
    if content.parts:
      all_parts.extend(content.parts)

  generated_images = gemini_generate_image.gemini_generate_images(
      parts=all_parts,
      aspect_ratio=parsed_aspect_ratio,
      model_name=model_name,
  )

  results = []
  for image_bytes, mime_type in generated_images:
    results.append(
        types.Content(
            parts=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type)]
        )
    )

  return results
