"""Tool for requesting input from user."""

from typing import Any, Dict
from opal_adk.types import valid_input
from google.adk.tools import tool_context as tc


async def chat_request_user_input(
    tool_context: tc.ToolContext,
    user_message: str,
    valid_input_type: valid_input.ValidInputTypes,
) -> Dict[str, Any]:
  """Requests input from user.

  Call this function to hold a conversation with the user. Each call
  corresponds to a conversation turn. Use only when necessary to fulfill the
  objective.

  Args:
    user_message: Message to display to the user when requesting input. The
      content may include references to VFS files using <file
      src="/vfs/name.ext" /> tags.
    valid_input_type: Input type hint, which allows to better present the chat
      user interface. If not specified, all kinds of inputs are accepted. When
      "text" is specified, the chat input is constrained to accept text only. If
      "file-upload" is specified, the input only allows uploading files. Unless
      the objective explicitly asks for a particular type of input, use the
      "any" value for "input_type" parameter, which does not constrain the
      input.

  Returns:
    A dictionary containing two keys, "requested_info" and "input_type".
  """
  # Tell the loop agent to stop looping. The user will need to respond 
  # before iteration continues. 
  tool_context.actions.escalate = True
  tool_context.actions.skip_summarization = True
  return {"requested_info": user_message, "input_type": valid_input_type}
