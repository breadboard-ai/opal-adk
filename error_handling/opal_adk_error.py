"""Opal ADK Errors messages."""

import json
import logging
import traceback

from google.genai import errors as genai_errors

from google.rpc import code_pb2


GENERIC_ERROR_MESSAGE = (
    ' An unexpected internal error occurred. Please try again.'
)
# Error when calling vertex.
GENERIC_MODEL_ERROR_MESSAGE = (
    ' An unexpected error occurred while calling the model. Please try again.'
)
GENERIC_CHAT_BASED_PREFIX = 'Sorry, I encountered an error. '


MODEL_CALL_ERROR_MESSAGE = (
    ' An unspecified error occurred while calling the model. Please try again.'
)

SAFETY_ERROR_MESSAGE = (
    ' Unable to generate response, generated content may violate safety'
    ' policies. Please try again with a different prompt.'
)

INVALID_IMAGE_FORMAT_ERROR_MESSAGE = (
    ' Provided image was not in a valid image format.'
)


class OpalAdkError(Exception):
  """Base class for Opal ADK errors."""

  def __init__(
      self,
      logged: str | None = None,
      status_message: str = GENERIC_ERROR_MESSAGE,
      status_code: code_pb2.Code = code_pb2.UNKNOWN,
      details: str = '',
      rewritten_intent: str = '',
  ):
    """Initializes the OpalAdkError.

    Args:
      logged: Error message to be logged. Defaults to status_message + details.
      status_message: The static human-readable "debug message" to be shown to
        the user. Must not contain any internal information. Should contain an
        actionable resolution to the error.
      status_code: The RPC error code.
      details: Any dynamic details about the error. Should not contain any
        internal information.
      rewritten_intent: Optional rewritten intent suggestion for validation
        errors.
    """
    if logged is None:
      logged = f'{status_message}'
      if details:
        logged += f'. Details: {details}'
    super().__init__(logged)
    self.status_message = status_message
    self.error_code = status_code
    self.details = details
    self.rewritten_intent = rewritten_intent

  def external_message(self) -> str:
    """Returns the error message to be shown to external users."""
    external_messages = {
        'code': code_pb2.Code.Name(self.error_code),
        'message': self.status_message,
        'details': self.details,
    }
    return json.dumps(external_messages)


class ChatError(OpalAdkError):
  """A wrapper for errors that need a chat based response."""

  def __init__(
      self,
      base_error: Exception,
      chat_prefix: str | None = None,
      full_chat_message: str | None = None,
  ):
    """Initializes the ChatError.

    This class takes any error and tries to create a chat based response. If the
    error has a full chat message, then we use that. Otherwise, if it's already
    an OpalAdkError, then we extract details about the error from it. If it's an
    unknown exception type, then we create a generic error for external users
    and a more detailed error for internal users.

    Args:
      base_error: The error to wrap. This should be an OpalAdkError, unknown
        errors will get a generic and unhelpful error message.
      chat_prefix: A prefix to prepend to the error message. Ignored if
        full_chat_message is provided.
      full_chat_message: The full chat message to return to the user. If not
        provided, then we will use chat_prefix or a generic prefix.
    """
    # Using get_opal_adk_error returns a generic error if the base_error is not
    # an OpalAdkError.
    error = get_opal_adk_error(base_error)
    if not chat_prefix:
      chat_prefix = GENERIC_CHAT_BASED_PREFIX

    chat_message = chat_prefix
    logged_message = f'Chat error: {chat_prefix}'
    if full_chat_message:
      logged_message += f'{full_chat_message}'
    else:
      if error.details:
        logged_message += f'Details: {error.details}. '
        chat_message += f'Details: {error.details}. '

    super().__init__(
        logged=logged_message,
        status_message=error.status_message,
        status_code=error.error_code,
        # The details field is shown to users
        details=full_chat_message or chat_message,
    )


def get_opal_adk_error(error: Exception) -> OpalAdkError:
  """Returns an OpalAdkError from the given exception."""
  full_traceback = traceback.format_exc()
  if isinstance(error, OpalAdkError):
    return error
  if isinstance(error, genai_errors.ClientError):
    if error.code == code_pb2.RESOURCE_EXHAUSTED:
      return OpalAdkError(
          logged=full_traceback,
          status_message=(
              'The system is experiencing higher load than usual. Please try'
              ' again later.'
          ),
          status_code=code_pb2.RESOURCE_EXHAUSTED,
      )
    else:
      return OpalAdkError(
          logged=full_traceback,
          status_message=GENERIC_MODEL_ERROR_MESSAGE,
          status_code=code_pb2.INTERNAL,
      )
  if isinstance(error, genai_errors.ServerError):
    return OpalAdkError(
        logged=full_traceback,
        status_message=GENERIC_MODEL_ERROR_MESSAGE,
        status_code=code_pb2.INTERNAL,
    )
  logging.info('Unhandled error (type %s): %s', type(error), error)
  # Return a generic error by default.
  return OpalAdkError(logged=full_traceback)


def get_error_as_chat_message(error: Exception) -> str:
  """Returns a chat message for the given error."""
  error = get_opal_adk_error(error)
  formatted_error = error.status_message + '\n' + error.details

  return formatted_error
