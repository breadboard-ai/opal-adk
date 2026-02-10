"""Client for interacting with Vertex AI using the genai library."""

from absl import logging
from google import genai
from opal_adk import flags


def create_vertex_ai_client(use_vertex: bool = False) -> genai.Client:
  """Creates a Vertex AI client using the genai library.

  The client is initialized with project and location from opal_adk flags.

  Args:
    use_vertex: If True the client will work with the Cloud Vertex API. If False
      it will use the Gemini API.  If using the Gemini API an API key will need
      to be provided in environmental variables. If Vertex AI API is being used
      a Google Cloud project and location will need to be provided in
      environmental variables.

  Returns:
    A genai.Client instance configured for Vertex AI.

  Raises:
    RuntimeError: If the genai.Client cannot be initialized.
  """
  try:
    vertex_client = genai.Client(
        vertexai=use_vertex,
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
