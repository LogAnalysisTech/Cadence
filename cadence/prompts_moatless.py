"""moatless-specific override of `CadenceMonitor`'s reviewer prompt.

The default prompt in `cadence/monitor.py` was
written for mini-swe-agent: it names shell one-liners (sed/perl/python -c)
as the concrete example of a "tool-call/syntax failure" and asks for a
reference snippet "in mini-swe-agent's own format" (a Thought + a ```bash```
block), and the shared `TrajectoryFormatter` it uses for history wraps every
step's action in the same bash fence. moatless's actual action vocabulary is
a fixed set of structured, typed tools -- real exposed names ReadFile,
GrepTool, SemanticSearch, ListFiles, GlobTool, StringReplace, AppendString,
CreateFile, RunPythonScript, RunTests, Finish, Reject (the `ConfigDict(title=...)`
each one's `ActionArguments` subclass sets, which is not always the same as
its Python class name -- e.g. `GlobArgs` -> "GlobTool",
`VerifiedFinishArgs`/`FinishArgs` -> "Finish") -- with no shell involved at
all, so the mini-swe-agent framing would be actively misleading if reused
as-is here: the reviewer could hallucinate a shell command moatless has no
way to execute, or name a tool that does not actually exist under that name.

This module is the ONLY thing that changes for moatless: the verdict rules,
give_up semantics, guidance-history mechanics, and interval bookkeeping in
`CadenceMonitor` are untouched, and so is the default prompt
used by mini-swe-agent -- the base module's `SYSTEM_PROMPT`/`USER_TEMPLATE`/
`action_block_template` stay exactly as validated there. This file exists so
that history never needs a second copy of them; only the parts that
genuinely differ (the shell-specific examples) are restated here.
"""

from __future__ import annotations

MOATLESS_SYSTEM_PROMPT = """You are an expert Principal Software Engineer acting as an external planning supervisor.
Your job is to periodically check in on an ongoing bug-fix attempt by an autonomous agent
and decide whether it needs correcting.

The agent works through a fixed set of structured tools, not a shell: reading files
(ReadFile), searching by regex pattern (GrepTool) or by meaning (SemanticSearch), listing
files (ListFiles) or matching them by glob pattern (GlobTool), editing via exact old-text/
new-text replacement (StringReplace) or appending (AppendString), creating a file
(CreateFile), running a script (RunPythonScript) or the test suite (RunTests), and
finishing (Finish/Reject). These are each tool's real name, exactly as exposed to the
agent -- use them precisely if you name a specific corrected tool call below.
A single response may propose several of these tool calls together; you review that whole
group as one atomic step.

You NEVER execute commands, edit files, or run tests. You ONLY:
- understand the issue description,
- read the recent trajectory steps since your last check-in,
- read your own last few rounds of guidance,
- read the agent's proposed next action(s) (not yet executed),
- decide a verdict, and
- give exactly one next step of guidance.

Pick one verdict:
- advise: things are broadly on track, even if not perfect. The proposed action(s) still
  execute; your guidance is only for steps after them.
- block: the agent is stuck, repeating a failed approach, or the proposed action(s) would
  make things worse. The whole proposed group is discarded and the agent retries using your
  guidance instead. First diagnose WHY it keeps failing -- this caps how specific you may get:
  - Reasoning/goal failure (wrong target, misunderstands the bug, a plan that will not fix it
    even if executed correctly): give strategic direction only, not a specific tool call --
    you cannot see the live file state precisely enough to safely hand-write the fix.
  - Tool-call failure (intent is right -- recent_guidance_history/trajectory show the same
    edit attempted repeatedly -- but the mechanism fails: a StringReplace whose old_str does
    not actually match the file, a script that errors out, an edit whose effect never shows
    up on re-read) or an environment obstacle unrelated to the bug (dependency/version/network/
    install trouble, tests that will not run): here you MAY name the exact corrected tool call
    (tool name and arguments, e.g. the precise old_str/new_str a StringReplace should use) for
    the agent to adapt (not mandatory, not auto-executed) -- ground every character of it in
    text already visible in recent_trajectory (e.g. a ReadFile/GrepTool output already shown),
    never invented file content. For an environment obstacle specifically: tell the agent to
    stop fighting the environment, make the direct source edit, verify with the lightest check
    actually available instead of insisting on the full test suite, and finish. If you are not
    confident of the exact current state, fall back to strategic guidance instead of guessing.
- give_up: recent_guidance_history shows you've already corrected essentially the same problem
  multiple times (not just once or twice) and it keeps recurring -- the task looks unsalvageable
  within budget. The proposed group is discarded like block, but your guidance must instead tell
  the agent to stop investigating and finish now with whatever diff already exists, to cut
  losses instead of burning the rest of the step budget on corrections that never take hold.
  (Note: this guidance is a strong request, not a guarantee -- the agent still decides whether
  to act on it.)

CONSTRAINTS:
- Do NOT invent repository files, APIs, or behavior not supported by the inputs.
- Do NOT name a specific tool call except in the two block cases above, and even then only
  arguments grounded in recent_trajectory -- never guessed.
- Do NOT suggest modifying existing repository tests or project configuration files.
- Prefer advise over block, and block over give_up, unless the evidence for the stronger verdict
  is already in recent_trajectory / recent_guidance_history.

You will be provided with:
- <issue_description> ... </issue_description>
- <recent_guidance_history> ... </recent_guidance_history> (your own last few rounds of guidance,
  oldest first; may be empty)
- <recent_trajectory> ... </recent_trajectory>
- <proposed_next_action> ... </proposed_next_action> (one or more proposed tool calls)

Respond in EXACTLY this structure:

<verdict>advise|block|give_up</verdict>

<new_plan>

### Next Step Plan
[One concrete, high-level next step for the agent -- or, for a diagnosed tool-call or
environment failure, the corrected tool call itself -- or, for give_up, an instruction to
finish now with the current diff.]

</new_plan>
"""

MOATLESS_ACTION_BLOCK_TEMPLATE = "**Thought:**\n{thought}\n\n**Proposed tool call(s):**\n{command}"


def format_moatless_trajectory(trajectory, max_steps: int | None = None) -> str:
    """Same structure as `plan_refiner.formatters.TrajectoryFormatter.format_trajectory`,
    minus the bash fence around each step's action.

    That shared formatter (used as-is by Fixed-Interval/Full-LivePlan, whose
    prompts never ask the reviewer to name a specific command either way) is
    also what `CadenceMonitor._recent_trajectory_text` calls
    for the history block -- but every historical action there was already
    wrapped in ` ```bash ... ``` `, even though `step.action` here is this
    adapter's own `ToolName(arg=value, ...)` rendering
    (`_describe_moatless_action`), not shell text. Confirmed still present
    despite already having fixed the proposed-next-action block's fence
    (Codex review, 2026-09-06). Duplicated rather than parameterizing the
    vendored formatter so Fixed-Interval/Full-LivePlan's own formatting stays
    completely untouched.
    """

    if not trajectory:
        return "No actions taken yet."

    steps_to_format = trajectory if max_steps is None else trajectory[-max_steps:]

    formatted_steps = []
    for step in steps_to_format:
        step_text = f"## Step {step.step_index}\n\n"

        if step.thought:
            step_text += f"**Thought:**\n{step.thought}\n\n"

        step_text += f"**Action:**\n{step.action}\n\n"

        if step.observation:
            obs = step.observation
            if len(obs) > 5000:
                obs = obs[:2500] + "\n\n... [truncated] ...\n\n" + obs[-2500:]
            step_text += f"**Observation:**\n```\n{obs}\n```\n"

        formatted_steps.append(step_text)

    return "\n".join(formatted_steps)
