"""Simple temporary main class to test that copybara is setup correctly.
"""

from collections.abc import Sequence

from absl import app


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  print('Hello world!')


if __name__ == "__main__":
  app.run(main)
