"""Data model for an Opal plan.

An OpalPlan is a collection of steps that define a sequence of operations
to be performed.
"""

import dataclasses
from opal_adk.data_model import opal_plan_step


@dataclasses.dataclass
class OpalPlan:
  """Represents a plan in an e2e Opal plan.

  An OpalPlan is a collection of steps that define a sequence of operations
  to be performed. Each step is typically a single Node in the Opal graph. A
  plan may contain a list of steps or a list of lists of steps. The latter is
  used to represent a plan that contains a list of steps that should be
  executed in parallel. Any plan step after the parrallel steps should be
  executed after all the parallel steps have completed.
  """

  plan_name: str
  plan_steps: dict[
      opal_plan_step.OpalPlanStep, list[opal_plan_step.OpalPlanStep]
  ]

  @property
  def render(self) -> str:
    """Renders the OpalPlan as a string."""

    def _render_step(
        step: opal_plan_step.OpalPlanStep, indent: int
    ) -> list[str]:
      """Renders a single step with the given indentation level."""
      raw_lines = step.render().strip().splitlines()
      if not raw_lines:
        return []

      rendered_lines = []
      rendered_lines.append(' ' * indent + raw_lines[0])

      inner_indent = ' ' * (indent + 2)
      for line in raw_lines[1:-1]:
        rendered_lines.append(inner_indent + line)

      if len(raw_lines) > 1:
        rendered_lines.append(' ' * indent + raw_lines[-1])

      return rendered_lines

    lines = []
    lines.append('<plan>')
    lines.append(f'  <plan_name>{self.plan_name}</plan_name>')
    lines.append('  <plan_steps>')

    for step_or_parallel_list in self.plan_steps:
      if isinstance(step_or_parallel_list, list):
        lines.append('    <parallel>')
        for step in step_or_parallel_list:
          lines.extend(_render_step(step, indent=6))
        lines.append('    </parallel>')
      else:
        lines.extend(_render_step(step_or_parallel_list, indent=4))

    lines.append('  </plan_steps>')
    lines.append('</plan>')

    return '\n'.join(lines)
