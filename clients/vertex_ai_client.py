"""Client for interacting with Vertex AI using the genai library."""

from absl import logging
from google import genai
from opal_adk import flags


def create_vertex_ai_client():
  """Creates a Vertex AI client using the genai library.

  The client is initialized with project and location from opal_adk flags.

  Returns:
    A genai.Client instance configured for Vertex AI.

  Raises:
    RuntimeError: If the genai.Client cannot be initialized.
  """
  try:
    vertex_client = genai.Client(
        vertexai=True,
        project=flags.get_project_id(),
        location=flags.get_location(),
    )
    logging.info("vertex_ai_client: Successfully created Vertex AI client.")
    return vertex_client
  except Exception as e:
    print(f"vertex_ai_client: Error initializing genai.Client: {e}")
    raise RuntimeError(
        "Could not initialize genai.Client. Set GOOGLE_CLOUD_PROJECT/LOCATION"
    ) from e
