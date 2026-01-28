import logging
from google.adk.agents import llm_agent
from google.adk.tools import agent_tool
from opal_adk.tools import fetch_url_contents_tool
from opal_adk.tools import map_search_tool
from opal_adk.tools import vertex_search_tool
from opal_adk.types import models
from opal_adk.types import output_type

_GENERATE_TEXT_INSTRUCTIONS = """You are working as part of an AI system, so
no chit-chat and no explaining what you're doing and why. DO NOT start with 
"Okay", or "Alright" or any preambles. Just the output, please."""


def generate_text(
    model: models.SimpleModel = models.SimpleModel.FLASH,
    output_format: output_type.OutputType = output_type.OutputType.TEXT,
    file_name: str | None = None,
    search_grounding: bool = False,
    maps_grounding: bool = False,
    url_context: bool = False,
) -> agent_tool.AgentTool:
  """An extremely versatile text generator, powered by Gemini.

  Use it for any tasks that involve generation of text. Supports multimodal
  content input.

  Args:
    model: ["pro", "flash", "lite"] The Gemini model to use for text generation.
      How to choose the right model: - choose "pro" when reasoning over complex
      problems in code, math, and STEM, as well as analyzing large datasets,
      codebases, and documents using long context. Use this model only when
      dealing with exceptionally complex problems. - choose "flash" for large
      scale processing, low-latency, high volume tasks that require thinking.
      This is the model you would use most of the time. - choose "lite" for high
      throughput. Use this model when speed is paramount.
    output_format: The output format. When "file" is specified, the output will
      be saved as a VFS file and the "file_path" response parameter will be
      provided as output. Use this when you expect a long output from the text
      generator. NOTE that choosing this option will prevent you from seeing the
      output directly: you only get back the VFS path to the file. You can read
      this file as a separate action, but if you do expect to read it, the
      "text" output format might be a better choice. When "text" is specified,
      the output will be returned as text directlty, and the "text" response
      parameter will be provided.`
    file_name: The name of the file to save the output to. This is the name that
      will come after "/vfs/" prefix in the file path. Use snake_case for
      naming. Only use when the "output_format" is set to "file".`
    search_grounding: Whether or not to use Google Search grounding. Grounding
      with Google Search connects the Gemini model to real-time web content and
      works with all available  languages. This allows Gemini to provide more
      accurate answers and cite verifiable sources beyond its knowledge cutoff.
    maps_grounding: Whether or not to use Google Maps grounding. Grounding with
      Google Maps connects the generative capabilities of Gemini with the rich,
      factual, and up-to-date data of Google Maps`
    url_context: Set to true to allow Gemini to retrieve context from URLs.
      Useful for tasks like the following: - Extract Data: Pull specific info
      like prices, names,or key findings from multiple URLs. - Compare
      documents, Using URLs, analyze multiple reports, articles, or PDFs to
      identify differences and track trends. - Synthesize & Create Content:
      Combine information from several source URLs to generate accurate
      summaries, blog posts, or reports. - Analyze Code & Docs: Point to a
      GitHub repository or technical documentation URL to explain code, generate
      setup instructions, or answer questions. Specify URLs in the prompt.
  Returns:
    An `AgentTool` instance configured for text generation with the specified
    grounding and context options.
  """
  logging.info("generate_text: output_format: %s", output_format)
  logging.info("generate_text: file_name: %s", file_name)
  filtered_tools = []
  if search_grounding:
    filtered_tools.append(vertex_search_tool.search_agent_tool())
  if maps_grounding:
    filtered_tools.append(map_search_tool.MapSearchTool)
  if url_context:
    filtered_tools.append(fetch_url_contents_tool.FetchUrlContentsTool)
  return agent_tool.AgentTool(
      llm_agent.LlmAgent(
          name="generate_text agent as a tool",
          description=(
              "An llm agent with specific instructions and tools for generating"
              " a plan to create text. This agent is used as a tool by other"
              " agents."
          ),
          instruction=_GENERATE_TEXT_INSTRUCTIONS,
          model=models.simple_model_to_model(model),  # pytype: disable=wrong-arg-types
          tools=filtered_tools,
      )
  )
