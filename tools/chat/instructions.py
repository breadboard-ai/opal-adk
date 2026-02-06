"""Module containing instructions and constants for the Opal ADK chat tool."""


CHAT_REQUEST_USER_INPUT = "chat_request_user_input"
CHAT_PRESENT_CHOICES = "chat_present_choices"
CHAT_LOG_VFS_PATH = "/vfs/system/chat_log.json"


CHAT_INSTRUCTIONS = f"""
## Interacting with the User
Use the "{CHAT_PRESENT_CHOICES}" function when you have a discrete set of options for the user 
to choose from. This provides a better user experience than asking them to type their selection.
Use the "{CHAT_REQUEST_USER_INPUT}" function for freeform text input or file uploads.
Prefer structured choices over freeform input when the answer space is bounded.
The chat log is maintained automatically at the VFS file "{CHAT_LOG_VFS_PATH}".
If the user input requires multiple entries, split the conversation into multiple 
turns. For example, if you have three questions to ask, ask them over three full conversation 
turns rather than in one call."""
