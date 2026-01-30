"""Utility functions for managing tools and agent functions."""

from collections.abc import Callable
from typing import Any, List, Tuple
from opal_adk.tools.generate import generate_text
from opal_adk.tools.generate import instructions as generate_instructions
from opal_adk.tools.system import instructions as system_instructions
from opal_adk.tools.system import objective_failed
from opal_adk.tools.system import objective_fulfilled
from opal_adk.types import model_constraint


def get_tools_with_model_constraints(
    constraint: model_constraint.ModelConstraint,
) -> List[Tuple[str, List[Callable[..., Any]]]]:
  match constraint:
    case (
        model_constraint.ModelConstraint.UNSPECIFIED
        | model_constraint.ModelConstraint.TEXT_FLASH
        | model_constraint.ModelConstraint.TEXT_PRO
        | model_constraint.ModelConstraint.IMAGE
        | model_constraint.ModelConstraint.VIDEO
        | model_constraint.ModelConstraint.SPEECH
        | model_constraint.ModelConstraint.MUSIC
    ):
      return [
          (
              system_instructions.SYSTEM_FUNCTIONS_INSTRUCTIONS,
              [
                  objective_failed.objective_failed,
                  objective_fulfilled.objective_fulfilled,
              ],
          ),
          (
              generate_instructions.GENERATE_INSTRUCTIONS,
              [generate_text.generate_text],
          ),
      ]
