from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional

from monitor_runtime.finding_lifecycle import FindingLifecycleConfig
from monitor_runtime.overhead import OverheadLedger

from monitor_runtime.types import BeforeActionResult

_VENDOR_DIR = Path(__file__).resolve().parents[1] / "monitor_baseline" / "vendor"
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

from plan_refiner.formatters import TrajectoryFormatter  # noqa: E402
from plan_refiner.models import OpenRouterModel  # noqa: E402
from plan_refiner.types import RefinerTrajectoryStep  # noqa: E402


BASE_INTERVAL = 5
MAX_INTERVAL = 20
GROWTH_STEP = 2
EXECUTION_MODES = {"severity", "always-advisory", "always-replacement"}
DEFAULT_MODEL = "gpt-5.2"

GUIDANCE_HISTORY_SIZE = 5

from .prompts import SYSTEM_PROMPT, USER_TEMPLATE

_VERDICT_RE = re.compile(r"<verdict>\s*(block|advise|give_up)\s*</verdict>", re.IGNORECASE)
_PLAN_RE = re.compile(r"<new_plan>(.*?)</new_plan>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class _CheckOutput:
    verdict: str
    verdict_confirmed: bool
    guidance: str
    raw_response: str

    @property
    def valid(self) -> bool:
        """A usable response: an explicit verdict tag and non-empty guidance.

        Anything else (missing/garbled verdict, missing or empty
        `<new_plan>`) is a format error -- the caller must not act on
        `verdict` or `guidance` for such a response.
        """

        return self.verdict_confirmed and bool(self.guidance)


def _parse_response(text: str) -> _CheckOutput:
    verdict_match = _VERDICT_RE.search(text)
    verdict = verdict_match.group(1).lower() if verdict_match else "advise"
    plan_match = _PLAN_RE.search(text)
    # No fallback to the raw text: without a matched `<new_plan>` block there
    # is no guidance, not "the whole response is the guidance" (that used to
    # let a bare `<verdict>block</verdict>` get injected as if it were advice).
    guidance = plan_match.group(1).strip() if plan_match else ""
    return _CheckOutput(
        verdict=verdict,
        verdict_confirmed=verdict_match is not None,
        guidance=guidance,
        raw_response=text,
    )


class CadenceMonitor:
    """Fixed-Interval baseline, extended so the refiner's own verdict drives
    both the block/advise decision and the check cadence.

    Same fixed-cadence, non-rule-engine shape as Fixed-Interval
    (`monitor_baseline_variant="refiner_only"`) and the same
    OpenRouter-backed supervisor call, but with these differences:
    - the check can block (discard the proposed action) instead of always
      being advisory-only;
    - it can also give_up on a run it judges unsalvageable, telling the
      agent to submit its current diff and stop instead of continuing to
      burn the step budget on corrections that keep not working;
    - the interval between checks is no longer fixed: every `advise`
      widens it by `GROWTH_STEP` up to `MAX_INTERVAL`; every `block` or
      `give_up` snaps it back to `BASE_INTERVAL`;
    - the last `GUIDANCE_HISTORY_SIZE` rounds of guidance (with their
      verdicts) are replayed on every check, so the model can recognize
      when it is repeating itself instead of judging each check in
      isolation.
    """

    is_null_monitor = False
    requires_agent_phase_reporting = False
    variant = "adaptive_interval"

    def __init__(self, *, refiner_model: Optional[str] = None, issue_description: str = "", fixed_interval: int = 0, base_interval: int = BASE_INTERVAL, growth_step: int = GROWTH_STEP, execution_mode: str = "severity") -> None:
        if type(fixed_interval) is not int or fixed_interval < 0:
            raise ValueError("fixed_interval must be a nonnegative integer (0 enables adaptive scheduling)")
        if type(base_interval) is not int or not 1 <= base_interval <= MAX_INTERVAL:
            raise ValueError("base_interval must be an integer between 1 and 20")
        if type(growth_step) is not int or growth_step < 1:
            raise ValueError("growth_step must be a positive integer")
        if execution_mode not in EXECUTION_MODES:
            raise ValueError(f"execution_mode must be one of {sorted(EXECUTION_MODES)}")
        self.execution_mode = execution_mode
        self.base_interval = base_interval
        self.growth_step = growth_step
        self.fixed_interval = fixed_interval
        model_name = (
            refiner_model
            or os.getenv("MONITOR_BASELINE_REFINER_MODEL")
            or DEFAULT_MODEL
        )
        self._model = OpenRouterModel(
            model_name=model_name,
            model_kwargs={
                "temperature": 0.0,
                "max_tokens": 3000,
                "extra_body": {"reasoning": {"effort": "medium"}},
            },
            set_cache_control="default_end",
        )
        self.overhead = OverheadLedger()
        # Mirrors LivePlanBoundaryMonitor: this baseline keeps its own state
        # and does not use the shared finding-lifecycle apparatus.
        self.finding_lifecycle = FindingLifecycleConfig(enabled=False)
        self.open_findings: tuple = ()
        self.issue_description = issue_description
        # Instance attributes (not just module constants) so a caller whose
        # agent has a different action vocabulary -- e.g. moatless's
        # structured tools instead of mini-swe-agent's shell commands -- can
        # swap in its own prompt after construction without subclassing or
        # touching this file. Left untouched, this is exactly the module
        # default: zero behavior change for existing callers.
        self.system_prompt = SYSTEM_PROMPT
        self.user_template = USER_TEMPLATE
        # Also overridable: wraps `command` in a bash fence, which only makes
        # sense when `command` really is a shell command (mini-swe-agent).
        # A caller with a structured/non-shell action vocabulary should
        # replace this along with `system_prompt`.
        self.action_block_template = "**Thought:**\n{thought}\n\n**Action:**\n```bash\n{command}\n```"
        # Also overridable, same reasoning: `TrajectoryFormatter.format_trajectory`
        # wraps every historical step's action in a bash fence too, not just
        # the proposed-next-action block above (Codex review, 2026-09-06).
        self.format_trajectory = TrajectoryFormatter.format_trajectory

        self._committed: list[RefinerTrajectoryStep] = []
        self._interval = self.fixed_interval or self.base_interval
        self._last_check_step: Optional[int] = None
        self._guidance_history: list[tuple[str, str]] = []
        self._executed_step_count = 0

    # -- accessors, mirroring LivePlanBoundaryMonitor's surface -----------

    @property
    def refiner_config(self) -> Optional[str]:
        return f"adaptive_interval (fixed interval={self.fixed_interval}, inline prompt)" if self.fixed_interval else f"adaptive_interval (base={self.base_interval}, growth={self.growth_step}, max={MAX_INTERVAL}, inline prompt)"

    @property
    def refiner_interval(self) -> int:
        return self._interval

    @property
    def executed_step_count(self) -> int:
        return self._executed_step_count

    # -- lifecycle ----------------------------------------------------------

    def reset_run_state(self, issue_description: str = "") -> None:
        self.overhead.reset()
        self.issue_description = issue_description
        self._committed = []
        self._interval = self.fixed_interval or self.base_interval
        self._last_check_step = None
        self._guidance_history = []
        self._executed_step_count = 0

    # -- cadence --------------------------------------------------------

    def _due_for_check(self, step_index: int) -> bool:
        if step_index < self._interval:
            return False
        if (
            self._last_check_step is not None
            and step_index - self._last_check_step < self._interval
        ):
            return False
        return True

    def _recent_trajectory_text(self) -> str:
        # Same "recent" scope as Fixed-Interval's refiner-only.yaml: full
        # trajectory on the first check, only steps since the last check
        # afterward. `_last_check_step` is the step_index of the *previous*
        # check (this method runs before the current check's own step_index
        # is recorded), and that step commits only after this check returns
        # -- so it lands in `_committed` for the *next* call. `>=` (not `>`)
        # is required so that step never gets silently dropped.
        if self._last_check_step is None:
            steps = self._committed
        else:
            steps = [s for s in self._committed if s.step_index >= self._last_check_step]
        return self.format_trajectory(steps)

    def _format_guidance_history(self) -> str:
        if not self._guidance_history:
            return "(none yet)"
        lines = []
        for i, (verdict, guidance) in enumerate(self._guidance_history, start=1):
            lines.append(f"{i}. [{verdict}] {guidance}")
        return "\n\n".join(lines)

    # -- pre-action -------------------------------------------------------

    def _next_interval(self, judged_verdict: str) -> int:
        if self.fixed_interval:
            return self.fixed_interval
        if judged_verdict == "advise":
            return min(MAX_INTERVAL, self._interval + self.growth_step)
        return self.base_interval

    def _execution_for(self, judged_verdict: str) -> str:
        if self.execution_mode == "always-advisory":
            return "advisory"
        if self.execution_mode == "always-replacement":
            return "replacement"
        return "advisory" if judged_verdict == "advise" else "replacement"

    def before_action(
        self,
        step_index: int,
        command: str,
        thought: str,
        *,
        accounting_step_id: Optional[str] = None,
    ) -> BeforeActionResult:
        accounting_step_id = accounting_step_id or str(step_index)
        if not self._due_for_check(step_index):
            return BeforeActionResult(execute=True, message=None, is_blocking=False)

        interval_before = self._interval
        output = self._invoke_check(
            step_index=step_index,
            command=command,
            thought=thought,
            accounting_step_id=accounting_step_id,
            interval_before=interval_before,
        )
        if not output.valid:
            # Malformed response (no verdict tag, or no/empty guidance): the
            # ledger already recorded a format_error event with the raw text
            # for offline analysis. Treat this check as if it never
            # happened -- no interval change, no guidance, action proceeds --
            # so a bad response can't be silently rewarded (longer interval)
            # or silently punished (a block with no actionable feedback).
            return BeforeActionResult(execute=True, message=None, is_blocking=False)

        self._last_check_step = step_index
        # Scheduling and guidance history always consume the original judgment.
        self._interval = self._next_interval(output.verdict)
        self._guidance_history.append((output.verdict, output.guidance))
        del self._guidance_history[:-GUIDANCE_HISTORY_SIZE]

        execution = self._execution_for(output.verdict)
        return BeforeActionResult(
            execute=execution == "advisory",
            message=f"[Adaptive Interval Guidance]\n\n{output.guidance}",
            is_blocking=execution == "replacement",
            # give_up asks the agent to submit and stop, but a text request
            # is not a guarantee -- force_submit tells the adapter to submit
            # the current diff itself instead of trusting compliance.
            # Uniform-execution ablations retain give_up text/reset, but
            # must not bypass their selected execution via forced submission.
            force_submit=output.verdict == "give_up" and self.execution_mode == "severity",
            judged_verdict=output.verdict,
            execution_mode=execution,
            execution_override=self.execution_mode,
        )

    def _invoke_check(
        self,
        *,
        step_index: int,
        command: str,
        thought: str,
        accounting_step_id: str,
        interval_before: int,
    ) -> _CheckOutput:
        user_prompt = self.user_template.format(
            issue_description=self.issue_description,
            recent_guidance_history=self._format_guidance_history(),
            recent_trajectory=self._recent_trajectory_text(),
            proposed_next_action=self.action_block_template.format(thought=thought, command=command),
        )
        started = perf_counter()
        try:
            text, usage = self._model.query(self.system_prompt, user_prompt)
        except Exception:
            self.overhead.record(
                component="monitor",
                operation="refiner_call",
                latency_seconds=perf_counter() - started,
                model=self._model.config.model_name,
                step_id=accounting_step_id,
                outcome="failed",
                metadata={"adaptive_interval_step_index": step_index, "interval_before": interval_before},
            )
            raise
        output = _parse_response(text)
        metadata: dict[str, object] = {
            "adaptive_interval_step_index": step_index,
            "verdict": output.verdict,
            "verdict_confirmed": output.verdict_confirmed,
            "guidance_valid": bool(output.guidance),
            "interval_before": interval_before,
            "judged_verdict": output.verdict if output.valid else None,
            "execution_override": self.execution_mode,
            "execution_mode": self._execution_for(output.verdict) if output.valid else None,
            "interval_after": self._next_interval(output.verdict) if output.valid else interval_before,
            "force_submit": output.valid and output.verdict == "give_up" and self.execution_mode == "severity",
        }
        if not output.valid:
            metadata["raw_response"] = output.raw_response
        self.overhead.record(
            component="monitor",
            operation="refiner_call",
            latency_seconds=perf_counter() - started,
            usage=usage,
            model=self._model.config.model_name,
            step_id=accounting_step_id,
            outcome="proposed" if output.valid else "format_error",
            metadata=metadata,
        )
        return output

    # -- post-action ------------------------------------------------------

    def after_action(
        self,
        step_index: int,
        command: str,
        thought: str,
        observation: str,
        executed: bool,
        *,
        accounting_step_id: Optional[str] = None,
    ) -> Optional[str]:
        if executed:
            self._committed.append(
                RefinerTrajectoryStep(
                    step_index=step_index,
                    thought=thought,
                    action=command,
                    observation=observation,
                )
            )
            self._executed_step_count += 1
        # Unlike Fixed-Interval's refiner_only variant, this baseline's check
        # runs pre-emptively in before_action (it needs to see the proposed
        # action to decide block/advise), so there is nothing left to do
        # post-action -- guidance was already returned synchronously.
        return None
