"""LLM Logging utility for beautiful, structured logging of model calls.

This module provides:
- LLMCallLog: Dataclass capturing complete model call data
- LLMTracer: Collects logs for an execution session with HAR export
- Context managers: log_operation, log_llm_call for easy integration
- Beautiful console rendering with box formatting and emojis
"""

import contextlib
import dataclasses
import datetime
import json
import logging
import threading
import time
from typing import Any

# Box drawing characters for beautiful console rendering
BOX_TOP_LEFT = '┌'
BOX_TOP_RIGHT = '┐'
BOX_BOTTOM_LEFT = '└'
BOX_BOTTOM_RIGHT = '┘'
BOX_HORIZONTAL = '─'
BOX_VERTICAL = '│'
BOX_CROSS_LEFT = '├'
BOX_CROSS_RIGHT = '┤'

# Double box for summary
DBOX_TOP_LEFT = '╔'
DBOX_TOP_RIGHT = '╗'
DBOX_BOTTOM_LEFT = '╚'
DBOX_BOTTOM_RIGHT = '╝'
DBOX_HORIZONTAL = '═'
DBOX_VERTICAL = '║'
DBOX_CROSS_LEFT = '╠'
DBOX_CROSS_RIGHT = '╣'

# Emojis for visual indicators
EMOJI_START = '🚀'
EMOJI_SUCCESS = '✅'
EMOJI_FAILURE = '❌'
EMOJI_PROMPT = '📝'
EMOJI_RESPONSE = '📄'
EMOJI_TOKENS = '📊'
EMOJI_THINKING = '🤔'

# Default box width
DEFAULT_BOX_WIDTH = 80
MAX_CONTENT_PREVIEW = 50000


@dataclasses.dataclass
class LLMCallLog:
  """Captures complete model call data."""

  operation_name: str
  timestamp: str
  model_name: str
  prompt: str
  latency_ms: float = 0.0
  system_instruction: str | None = None
  config: dict[str, Any] | None = None
  response_text: str | None = None
  response_schema: str | None = None
  token_usage: dict[str, int] | None = None
  status: str = 'pending'
  error_message: str | None = None
  metadata: dict[str, Any] | None = None
  depth: int = 0
  thinking_text: str | None = None

  def to_dict(self) -> dict[str, Any]:
    """Convert to dictionary for JSON serialization."""
    return {
        'operation_name': self.operation_name,
        'timestamp': self.timestamp,
        'model_name': self.model_name,
        'prompt': self.prompt,
        'system_instruction': self.system_instruction,
        'config': self.config,
        'response_text': self.response_text,
        'response_schema': self.response_schema,
        'latency_ms': self.latency_ms,
        'token_usage': self.token_usage,
        'status': self.status,
        'error_message': self.error_message,
        'metadata': self.metadata,
    }

  def to_har_entry(self) -> dict[str, Any]:
    """Convert to HAR-like entry format."""
    headers = []
    if self.config:
      for key, value in self.config.items():
        if value is not None:
          headers.append({'name': str(key), 'value': str(value)})

    return {
        'startedDateTime': self.timestamp,
        'time': self.latency_ms,
        'request': {
            'method': 'POST',
            'url': self.model_name,
            'headers': headers,
            'postData': {
                'mimeType': 'text/plain',
                'text': self.prompt,
            },
        },
        'response': {
            'status': 200 if self.status == 'success' else 500,
            'statusText': self.status,
            'content': {
                'mimeType': (
                    'application/json' if self.response_schema else 'text/plain'
                ),
                'text': self.response_text or '',
                'size': len(self.response_text) if self.response_text else 0,
            },
        },
        'timings': {
            'wait': 0,
            'receive': self.latency_ms,
        },
        'comment': self.operation_name,
    }


class LLMTracer:
  """Collects logs for an execution session with HAR export support."""

  def __init__(self, session_name: str = 'default'):
    self.session_name = session_name
    self.logs: list[LLMCallLog] = []
    self._start_time = time.time()
    self.current_depth = 0

  def add_log(self, log: LLMCallLog) -> None:
    """Add a log entry."""
    self.logs.append(log)

  def get_logs(self) -> list[LLMCallLog]:
    """Return all collected logs."""
    return self.logs

  def clear(self) -> None:
    """Clear all logs."""
    self.logs = []
    self._start_time = time.time()

  def export_har(self) -> dict[str, Any]:
    """Export logs in HAR-like format."""
    return {
        'log': {
            'version': '1.0',
            'creator': {
                'name': 'intent2app',
                'version': '1.0',
            },
            'entries': [log.to_har_entry() for log in self.logs],
        }
    }

  def export_json(self) -> str:
    """Export logs as JSON string."""
    return json.dumps(
        {
            'session_name': self.session_name,
            'total_duration_ms': (time.time() - self._start_time) * 1000,
            'logs': [log.to_dict() for log in self.logs],
        },
        indent=2,
    )

  def render_summary(self) -> str:
    """Return pretty-printed summary table."""
    if not self.logs:
      return 'No LLM calls recorded.'

    lines = []
    width = DEFAULT_BOX_WIDTH

    # Header
    lines.append(
        f'{DBOX_TOP_LEFT}{DBOX_HORIZONTAL * (width - 2)}{DBOX_TOP_RIGHT}'
    )
    title = 'LLM CALL SUMMARY'
    padding = (width - 2 - len(title)) // 2
    lines.append(
        f'{DBOX_VERTICAL}{" " * padding}{title}{" " * (width - 2 - padding - len(title))}{DBOX_VERTICAL}'
    )
    lines.append(
        f'{DBOX_CROSS_LEFT}{DBOX_HORIZONTAL * (width - 2)}{DBOX_CROSS_RIGHT}'
    )

    # Column headers
    header = (
        f'{DBOX_VERTICAL} {"Operation":<30} {BOX_VERTICAL} {"Model":<14}'
        f' {BOX_VERTICAL} {"Time":>6} {BOX_VERTICAL} {"Tokens":>10}'
        f' {DBOX_VERTICAL}'
    )
    lines.append(header)
    lines.append(
        f'{DBOX_CROSS_LEFT}{BOX_HORIZONTAL * 32}{BOX_VERTICAL}{BOX_HORIZONTAL * 16}{BOX_VERTICAL}{BOX_HORIZONTAL * 8}{BOX_VERTICAL}{BOX_HORIZONTAL * 12}{DBOX_CROSS_RIGHT}'
    )

    # Data rows
    total_time = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    for log in self.logs:
      op_name = log.operation_name[:30]
      model = log.model_name.split('/')[-1][:14] if log.model_name else 'N/A'
      time_str = f'{log.latency_ms / 1000:.2f}s'
      total_time += log.latency_ms

      token_usage = log.token_usage
      if token_usage is not None:
        input_tokens = token_usage.get('input_tokens', 0)
        output_tokens = token_usage.get('output_tokens', 0)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        tokens_str = f'{input_tokens}/{output_tokens}'
      else:
        tokens_str = 'N/A'

      row = (
          f'{DBOX_VERTICAL} {op_name:<30} {BOX_VERTICAL} {model:<14}'
          f' {BOX_VERTICAL} {time_str:>6} {BOX_VERTICAL} {tokens_str:>10}'
          f' {DBOX_VERTICAL}'
      )
      lines.append(row)

    # Total row
    lines.append(
        f'{DBOX_CROSS_LEFT}{DBOX_HORIZONTAL * (width - 2)}{DBOX_CROSS_RIGHT}'
    )
    total_row = (
        f'{DBOX_VERTICAL} {"TOTAL":<30} {BOX_VERTICAL} '
        f'{len(self.logs)} calls{" " * 8} {BOX_VERTICAL} '
        f'{total_time / 1000:.2f}s {BOX_VERTICAL} '
        f'{total_input_tokens}/{total_output_tokens}{" " * 2} {DBOX_VERTICAL}'
    )
    lines.append(total_row)

    # Footer
    lines.append(
        f'{DBOX_BOTTOM_LEFT}{DBOX_HORIZONTAL * (width - 2)}{DBOX_BOTTOM_RIGHT}'
    )

    return '\n'.join(lines)


# Thread-local storage for current tracer
_thread_local = threading.local()


def get_current_tracer() -> LLMTracer:
  """Get the current thread's tracer, creating one if necessary."""
  if not hasattr(_thread_local, 'tracer'):
    _thread_local.tracer = LLMTracer()
  return _thread_local.tracer


def set_current_tracer(tracer: LLMTracer) -> None:
  """Set the current thread's tracer."""
  _thread_local.tracer = tracer


def set_content_suppression(suppress: bool) -> None:
  """Set whether to suppress LLM content (prompts, responses) in logs.

  Uses thread-local storage, so each RPC request has isolated state.
  This is safe to call at the start of each RPC handler.

  Args:
    suppress: If True, prompt/response content will not be logged.
  """
  _thread_local.suppress_content = suppress


def is_content_suppressed() -> bool:
  """Check if LLM content logging is currently suppressed.

  Returns:
    True if content should be suppressed, False otherwise.
  """
  return getattr(_thread_local, 'suppress_content', False)


def _get_timestamp() -> str:
  """Get current timestamp in ISO 8601 format."""
  return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _render_box(
    title: str,
    content: str,
    width: int = DEFAULT_BOX_WIDTH,
    indent: int = 0,
) -> str:
  """Render content in a box with title."""
  indent_str = '  ' * indent
  lines = []
  inner_width = width - 4  # Account for box chars and padding

  # Top border with title
  title_line = f' {title} '
  remaining = inner_width - len(title_line)
  left_pad = remaining // 2
  right_pad = remaining - left_pad
  lines.append(
      f'{indent_str}{BOX_TOP_LEFT}{BOX_HORIZONTAL * left_pad}{title_line}{BOX_HORIZONTAL * right_pad}{BOX_TOP_RIGHT}'
  )

  # Content lines
  for line in content.split('\n'):
    lines.append(f'{indent_str}{BOX_VERTICAL} {line} {BOX_VERTICAL}')

  # Bottom border
  lines.append(
      f'{indent_str}{BOX_BOTTOM_LEFT}{BOX_HORIZONTAL * (width - 2)}{BOX_BOTTOM_RIGHT}'
  )

  return '\n'.join(lines)


# Track start times for log_operation_start/log_operation_end pairs
_operation_start_times: dict[str, float] = {}


def log_operation_start(operation_name: str) -> None:
  """Log the start of an operation (non-context-manager version).

  Use this with log_operation_end when a context manager is not suitable
  (e.g., functions with multiple return paths or complex control flow).

  This function is designed to never raise exceptions.

  Args:
    operation_name: Human-readable name of the operation.
  """

  try:
    tracer = get_current_tracer()
    time_str = datetime.datetime.now().strftime('%H:%M:%S')
    _log_styled(
        f'{EMOJI_START} [{time_str}] Starting: {operation_name}',
        indent=tracer.current_depth,
    )
    _operation_start_times[operation_name] = time.time()
    tracer.current_depth += 1
  except Exception as e:  # pylint: disable=broad-except
    logging.warning('log_operation_start failed: %s', e)


def log_operation_end(operation_name: str, success: bool = True) -> None:
  """Log the end of an operation (non-context-manager version).

  This function is designed to never raise exceptions.

  Args:
    operation_name: Human-readable name of the operation (must match start).
    success: Whether the operation succeeded.
  """
  try:
    tracer = get_current_tracer()
    tracer.current_depth = max(0, tracer.current_depth - 1)

    start_time = _operation_start_times.pop(operation_name, time.time())
    elapsed = time.time() - start_time
    time_str = datetime.datetime.now().strftime('%H:%M:%S')

    if success:
      _log_styled(
          f'{EMOJI_SUCCESS} [{time_str}] Completed: {operation_name}'
          f' ({elapsed:.2f}s)',
          indent=tracer.current_depth,
      )
    else:
      _log_styled(
          f'{EMOJI_FAILURE} [{time_str}] Failed: {operation_name}'
          f' ({elapsed:.2f}s)',
          indent=tracer.current_depth,
      )
  except Exception as e:  # pylint: disable=broad-except
    logging.warning('log_operation_end failed: %s', e)


def _log_styled(message: str, indent: int = 0) -> None:
  """Log a styled message with indentation."""
  indent_str = '  ' * indent
  logging.info('%s%s', indent_str, message)


@contextlib.contextmanager
def log_operation(operation_name: str):
  """Context manager for logging operation start/end with timing.

  This context manager is designed to never break the main application flow
  due to logging failures. User code exceptions are still re-raised.

  Args:
    operation_name: Human-readable name of the operation being logged.

  Yields:
    None. The context manager handles logging automatically.

  Raises:
    Exception: Re-raises any exception from the wrapped code.

  Usage:
    with log_operation('Generate High Level Plan'):
      plan = planner.generate_high_level_plan(...)
  """
  # Setup with fallback for logging failures
  try:
    tracer = get_current_tracer()
    depth = tracer.current_depth
    tracer.current_depth += 1
    timestamp = _get_timestamp()
    start_time = time.time()
    time_str = datetime.datetime.now().strftime('%H:%M:%S')
    _log_styled(
        f'{EMOJI_START} [{time_str}] Starting: {operation_name}', indent=depth
    )
    log = LLMCallLog(
        operation_name=operation_name,
        timestamp=timestamp,
        model_name='',
        prompt='',
        depth=depth,
    )
  except Exception as e:  # pylint: disable=broad-except
    logging.warning('log_operation setup failed: %s', e)
    # Provide minimal fallback values
    tracer = None
    depth = 0
    start_time = time.time()
    log = None

  user_exception = None
  try:
    yield
  except Exception as e:  # pylint: disable=broad-except
    user_exception = e

  # Cleanup and logging with fallback for logging failures
  try:
    elapsed_ms = (time.time() - start_time) * 1000
    if log:
      log.latency_ms = elapsed_ms
      if user_exception:
        log.status = 'error'
        log.error_message = str(user_exception)
      else:
        log.status = 'success'

    time_str = datetime.datetime.now().strftime('%H:%M:%S')
    if user_exception:
      _log_styled(
          f'{EMOJI_FAILURE} [{time_str}] Failed: {operation_name}'
          f' ({elapsed_ms / 1000:.2f}s) - {user_exception}',
          indent=depth,
      )
    else:
      _log_styled(
          f'{EMOJI_SUCCESS} [{time_str}] Completed: {operation_name}'
          f' ({elapsed_ms / 1000:.2f}s)',
          indent=depth,
      )
  except Exception as e:  # pylint: disable=broad-except
    logging.warning('log_operation cleanup failed: %s', e)

  # Always try to restore tracer state and add log
  try:
    if tracer:
      tracer.current_depth -= 1
      if log:
        tracer.add_log(log)
  except Exception as e:  # pylint: disable=broad-except
    logging.warning('log_operation finalization failed: %s', e)

  # Re-raise user exception if any
  if user_exception:
    raise user_exception


class LLMCallContext:
  """Context object for log_llm_call that allows setting response data."""

  def __init__(self, log: LLMCallLog, tracer: LLMTracer):
    self._log = log
    self._tracer = tracer

  def set_response(
      self,
      response_text: str | None,
      usage_metadata: Any = None,
      thinking_text: str | None = None,
  ) -> None:
    """Set the response text, token usage, and thinking text.

    This method is designed to never raise exceptions, ensuring that logging
    failures cannot break the main application flow.

    Args:
      response_text: The response text from the model.
      usage_metadata: Token usage metadata from the model response.
      thinking_text: Thinking tokens extracted from the model response.
    """

    try:
      self._log.response_text = response_text
      self._log.thinking_text = thinking_text
      if usage_metadata:
        # Handle google.genai.types.GenerateContentResponseUsageMetadata
        if hasattr(usage_metadata, 'prompt_token_count'):
          self._log.token_usage = {
              'input_tokens': getattr(usage_metadata, 'prompt_token_count', 0),
              'output_tokens': getattr(
                  usage_metadata, 'candidates_token_count', 0
              ),
              'total_tokens': getattr(usage_metadata, 'total_token_count', 0),
              'thoughts_tokens': getattr(
                  usage_metadata, 'thoughts_token_count', 0
              ),
          }
        elif isinstance(usage_metadata, dict):
          self._log.token_usage = usage_metadata
    except Exception as e:  # pylint: disable=broad-except
      logging.warning('Failed to set response in logging context: %s', e)


@contextlib.contextmanager
def log_llm_call(
    operation_name: str,
    model_name: str,
    prompt: str,
    system_instruction: str | None = None,
    config: dict[str, Any] | None = None,
    response_schema: str | None = None,
    metadata: dict[str, Any] | None = None,
):
  """Context manager for logging LLM calls with prompts and responses.

  This context manager is designed to never break the main application flow
  due to logging failures. User code exceptions are still re-raised.

  Args:
    operation_name: Human-readable name of the LLM operation.
    model_name: Model identifier (e.g., 'gemini-2.5-flash').
    prompt: The prompt text sent to the model.
    system_instruction: Optional system instruction for the model.
    config: Optional model configuration dict (temperature, max_tokens, etc.).
    response_schema: Optional name of the expected response schema.
    metadata: Optional additional context metadata.

  Yields:
    LLMCallContext: Context object with set_response() method for capturing
      the response text and token usage after the LLM call completes.

  Raises:
    Exception: Re-raises any exception from the wrapped code.

  Usage:
    with log_llm_call(
        operation_name='Generate AppConfig',
        model_name='gemini-2.5-flash',
        prompt=rendered_prompt,
        response_schema='AppConfig',
    ) as ctx:
      response = client.generate_content(...)
      ctx.set_response(response.text, response.usage_metadata)
  """
  # Setup with fallback for logging failures
  try:
    tracer = get_current_tracer()
    depth = tracer.current_depth
    tracer.current_depth += 1
    timestamp = _get_timestamp()
    start_time = time.time()
    time_str = datetime.datetime.now().strftime('%H:%M:%S')

    log = LLMCallLog(
        operation_name=operation_name,
        timestamp=timestamp,
        model_name=model_name,
        prompt=prompt,
        system_instruction=system_instruction,
        config=config,
        response_schema=response_schema,
        metadata=metadata,
        depth=depth,
    )

    # Log start with beautiful formatting
    model_short = model_name.split('/')[-1] if model_name else 'unknown'
    header_line = f'{EMOJI_START} [{time_str}] Starting: {operation_name}'
    _log_styled(header_line, indent=depth)

    config_parts = [f'Model: {model_short}']
    if config and 'temperature' in config:
      config_parts.append(f"Temp: {config['temperature']}")
    if response_schema:
      config_parts.append(f'Schema: {response_schema}')
    _log_styled(f'  {" | ".join(config_parts)}', indent=depth)

    # Log full prompt (unless suppressed for Dasher users)
    if not is_content_suppressed():
      _log_styled(f'{EMOJI_PROMPT} PROMPT ({len(prompt)} chars):', indent=depth)
      for line in prompt.split('\n'):
        _log_styled(f'  {line}', indent=depth)

    ctx = LLMCallContext(log, tracer)
  except Exception as e:  # pylint: disable=broad-except
    logging.warning('log_llm_call setup failed: %s', e)
    # Provide minimal fallback
    tracer = None
    depth = 0
    start_time = time.time()
    log = None
    ctx = None

  # If setup failed, create a dummy context
  if ctx is None:
    ctx = type('DummyCtx', (), {'set_response': lambda *a, **k: None})()

  user_exception = None
  try:
    yield ctx
  except Exception as e:  # pylint: disable=broad-except
    user_exception = e

  # Cleanup and logging with fallback for logging failures
  try:
    elapsed_ms = (time.time() - start_time) * 1000
    if log:
      log.latency_ms = elapsed_ms
      if user_exception:
        log.status = 'error'
        log.error_message = str(user_exception)
      else:
        log.status = 'success'

      # Log full response (unless suppressed for Dasher users)
      if log.response_text and not is_content_suppressed():
        _log_styled(
            f'{EMOJI_RESPONSE} RESPONSE ({len(log.response_text)} chars):',
            indent=depth,
        )
        for line in log.response_text.split('\n'):
          _log_styled(f'  {line}', indent=depth)

      # Log thinking tokens (unless suppressed for Dasher users)
      if log.thinking_text and not is_content_suppressed():
        _log_styled(
            f'{EMOJI_THINKING} THINKING ({len(log.thinking_text)} chars):',
            indent=depth,
        )
        for line in log.thinking_text.split('\n'):
          _log_styled(f'  {line}', indent=depth)

      # Log token usage
      if log.token_usage:
        input_t = log.token_usage.get('input_tokens', 0)
        output_t = log.token_usage.get('output_tokens', 0)
        thoughts_t = log.token_usage.get('thoughts_tokens', 0)
        token_str = f'{EMOJI_TOKENS} Tokens: {input_t} in / {output_t} out'
        if thoughts_t > 0:
          token_str += f' / {thoughts_t} thoughts'
        _log_styled(token_str, indent=depth)

    time_str = datetime.datetime.now().strftime('%H:%M:%S')
    if user_exception:
      _log_styled(
          f'{EMOJI_FAILURE} [{time_str}] Failed: {operation_name}'
          f' ({elapsed_ms / 1000:.2f}s) - {user_exception}',
          indent=depth,
      )
    else:
      _log_styled(
          f'{EMOJI_SUCCESS} [{time_str}] Completed: {operation_name}'
          f' ({elapsed_ms / 1000:.2f}s)',
          indent=depth,
      )
  except Exception as e:  # pylint: disable=broad-except
    logging.warning('log_llm_call cleanup failed: %s', e)

  # Always try to restore tracer state and add log
  try:
    if tracer:
      tracer.current_depth -= 1
      if log:
        tracer.add_log(log)
  except Exception as e:  # pylint: disable=broad-except
    logging.warning('log_llm_call finalization failed: %s', e)

  # Re-raise user exception if any
  if user_exception:
    raise user_exception


def print_summary() -> None:
  """Print the summary table for the current session."""
  tracer = get_current_tracer()
  print(tracer.render_summary())


def export_har_to_file(filepath: str) -> None:
  """Export the current session's logs to a HAR file."""
  tracer = get_current_tracer()
  har_data = tracer.export_har()
  with open(filepath, 'w') as f:
    json.dump(har_data, f, indent=2)
  logging.info('Exported HAR to %s', filepath)
