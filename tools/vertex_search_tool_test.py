"""Tests for opal_adk.tools.vertex_search_tool."""

import unittest
from unittest import mock

from google import genai
from google.adk.tools import tool_context
from google.genai import types
from opal_adk.tools import vertex_search_tool
from opal_adk.types import models


class VertexSearchToolTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_genai_client = mock.MagicMock(spec=genai.Client)
    self.tool = vertex_search_tool.VertexSearchTool(
        genai_client=self.mock_genai_client
    )

  def test_init_defaults(self):
    self.assertEqual(self.tool.name, "OpalAdkVertexSearchTool")
    self.assertEqual(
        self.tool.description, "Performs a web search using Vertex AI."
    )
    self.assertEqual(
        self.tool.model, models.Models.FLASH_MODEL_NAME.value
    )
    self.assertIsNone(self.tool.text_safety_settings)

  def test_init_custom(self):
    safety_settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        )
    ]
    tool = vertex_search_tool.VertexSearchTool(
        genai_client=self.mock_genai_client,
        text_safety_settings=safety_settings,
        model=models.Models.LITE_MODEL_NAME,
    )
    self.assertEqual(tool.model, models.Models.LITE_MODEL_NAME.value)
    self.assertEqual(tool.text_safety_settings, safety_settings)

  @mock.patch("opal_adk.util.gemini_utils.extract_grounding_metadata")
  def test_call_successful(self, mock_extract_metadata):
    query = "test query"
    mock_context = mock.MagicMock(spec=tool_context.ToolContext)

    mock_response = mock.MagicMock(spec=types.GenerateContentResponse)
    mock_response.text = "Test search result"
    mock_response.candidates = [mock.MagicMock()]
    self.mock_genai_client.models.generate_content.return_value = mock_response

    expected_metadata = [types.Content()]
    mock_extract_metadata.return_value = expected_metadata

    result = self.tool(query=query, context=mock_context)

    self.assertEqual(
        result,
        {
            "result": "Test search result",
            "grounding_metadata:": expected_metadata,
        },
    )

    self.mock_genai_client.models.generate_content.assert_called_once()
    _, kwargs = self.mock_genai_client.models.generate_content.call_args
    self.assertEqual(
        kwargs["model"], models.Models.GEMINI_2_5_FLASH.value
    )
    self.assertEqual(kwargs["contents"], query)
    config = kwargs["config"]
    self.assertIsInstance(config.tools[0].google_search, types.GoogleSearch)

    mock_extract_metadata.assert_called_once_with(mock_response)

  def test_call_no_candidates(self):
    query = "test query"
    mock_context = mock.MagicMock(spec=tool_context.ToolContext)

    mock_response = mock.MagicMock(spec=types.GenerateContentResponse)
    mock_response.text = "Test search result"
    mock_response.candidates = []
    self.mock_genai_client.models.generate_content.return_value = mock_response

    with mock.patch(
        "opal_adk.util.gemini_utils.extract_grounding_metadata"
    ) as mock_extract_metadata:
      result = self.tool(query=query, context=mock_context)

      self.assertEqual(result, {"result": []})
      mock_extract_metadata.assert_not_called()


if __name__ == "__main__":
  unittest.main()
