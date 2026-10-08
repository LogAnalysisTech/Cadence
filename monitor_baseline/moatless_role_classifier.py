"""Maps moatless `ActionArguments` instances directly to the LivePlan role
vocabulary (`L_navigate`/`L_reproduce`/`P`/`V_newly_generated_test`/
`V_regression_test`/`general`), bypassing `plan_monitor`'s
`CommandParser`/`mapLang.get_action_role` entirely.

Moatless does not fake bash text through the swe-agent-shaped parser --
each `ActionArguments` subclass already carries structured fields (a file
path, a test-file list, a bash command string, ...), so this module inspects
those fields directly using the same semantics as
`monitor_baseline/vendor/plan_monitor/mapLang.py`'s docstring:

- L_navigate: read-only browsing/searching of non-test files.
- L_reproduce: generating/viewing/executing tests, before the first patch.
- P: creating/editing/deleting non-test assets.
- V_newly_generated_test / V_regression_test: any interaction with tests
  after the first patch, split by whether the test path was newly created
  this run.
- general: everything else (pure reasoning, task-list bookkeeping, ...).

`Finish`/`VerifiedFinish`/`Reject` map to a dedicated ending signal (checked
via `is_ending_action`), not a bare `general` role -- this mirrors how
`plan_monitor/monitor.py`'s STEP-1 special-cases SWE-agent's own `submit`
command to trigger the plan-compliance rule's "ending flags" branch even
though `submit` itself classifies as `general`.
"""

from __future__ import annotations

from typing import Any, Optional, Set

TEST_HINTS = ("test_", "repro", "reproduc", "debug", "_test", "/tests/", "/test/")


def _is_test_path(path: Optional[str]) -> bool:
    if not path:
        return False
    lowered = path.lower()
    return any(hint in lowered for hint in TEST_HINTS)


# Action classes whose only job is to end the run -- the plan-compliance
# "ending flags" equivalent for moatless. Not `general`; use `is_ending_action`.
ENDING_CLASSES = {"FinishArgs", "VerifiedFinishArgs", "RejectArgs"}

# Read-only browsing/searching -- always L_navigate regardless of prior patch state.
NAVIGATE_CLASSES = {
    "GlobArgs",
    "ListFilesArgs",
    "GrepToolArgs",
    "ReadFilesArgs",
    "ViewDiffArgs",
    "FindCodeSnippetArgs",
    "FindFunctionArgs",
    "FindClassArgs",
    "SemanticSearchArgs",
    "CleanupContextArgs",
    "ListTasksArgs",
}

# Pure reasoning / bookkeeping / communication -- no phase role.
GENERAL_CLASSES = {"RespondArgs", "ThinkArgs"}

# Non-test-aware edits: always P (no file-path field to check for test-ness).
PATCH_CLASSES = {
    "AddCodingTasksArgs",
    "CreateTasksArgs",
    "CreateJsonArtifactArgs",
    "FinishCodingTasksArgs",
    "DefineAPIArgs",
    "RemoveCodingTasksArgs",
    "UpdateTaskArgs",
}

# Edits with a single target `path` field: P unless the path looks like a test,
# in which case treat like the str_replace_editor test-provenance rule.
PATH_BEARING_EDIT_CLASSES = {
    "AppendStringArgs": "path",
    "CreatePythonFileArgs": "path",
    "CreateFileArgs": "path",
    "InsertLinesArgs": "path",
    "StringReplaceArgs": "path",
}

# Test-execution actions: L_reproduce before first patch, V_* after.
TEST_RUN_LIST_FIELDS = {
    "RunTestsArgs": "test_files",
    "RunMavenTestsArgs": "test_files",
    "RunMyPyArgs": "files",
}
TEST_RUN_SCALAR_FIELDS = {
    "RunPythonScriptArgs": "script_path",
}
# No path at all -- ephemeral/inline validation code, same as mapLang's
# "dynamic suite" handling for `python -c`/`python -m` without on-disk paths.
TEST_RUN_DYNAMIC_CLASSES = {"RunPythonCodeArgs"}

EDIT_ACTION_EDIT_COMMANDS = {"create", "str_replace", "insert", "undo_edit"}


def is_ending_action(action: Any) -> bool:
    """True for moatless's submit-equivalent actions (Finish/VerifiedFinish/Reject)."""

    return type(action).__name__ in ENDING_CLASSES


def classify_moatless_action(
    action: Any,
    *,
    has_patch: bool,
    created_tests: Optional[Set[str]] = None,
) -> str:
    """Classify one moatless `ActionArguments` instance into a LivePlan role.

    `has_patch`: whether a `P` action has already occurred earlier in this
    node's root-to-node path (mirrors `mapLang._has_prior_patch`).
    `created_tests`: mutable set of test paths created earlier in this run;
    updated in place when a new test file is created here (mirrors
    `StatefulPhaseMonitor.created_tests`).
    """

    cls_name = type(action).__name__

    if cls_name in ENDING_CLASSES:
        # Ending is signaled separately via `is_ending_action`; the role
        # itself does not participate in phase tracking.
        return "general"

    if cls_name in NAVIGATE_CLASSES:
        return "L_navigate"

    if cls_name in GENERAL_CLASSES:
        return "general"

    if cls_name in PATCH_CLASSES:
        return "P"

    if cls_name == "BashArgs":
        command = str(getattr(action, "command", "") or "")
        if _is_test_path(command) or "pytest" in command.lower():
            return _post_or_pre_patch(command, has_patch, created_tests)
        return "general"

    if cls_name == "EditActionArguments":
        command = str(getattr(action, "command", "") or "").lower()
        path = getattr(action, "path", None)
        if command == "view":
            if _is_test_path(path):
                return _post_or_pre_patch(path, has_patch, created_tests)
            return "L_navigate"
        if command in EDIT_ACTION_EDIT_COMMANDS:
            if _is_test_path(path):
                if command == "create" and created_tests is not None and path:
                    created_tests.add(path)
                return _post_or_pre_patch(path, has_patch, created_tests)
            return "P"
        return "general"

    if cls_name in PATH_BEARING_EDIT_CLASSES:
        field = PATH_BEARING_EDIT_CLASSES[cls_name]
        path = getattr(action, field, None)
        if _is_test_path(path):
            if created_tests is not None and path:
                created_tests.add(path)
            return _post_or_pre_patch(path, has_patch, created_tests)
        return "P"

    if cls_name == "ReadFileArgs":
        path = getattr(action, "file_path", None)
        if _is_test_path(path):
            return _post_or_pre_patch(path, has_patch, created_tests)
        return "L_navigate"

    if cls_name == "ViewCodeArgs":
        spans = getattr(action, "files", None) or []
        paths = [getattr(span, "file_path", None) for span in spans]
        test_paths = [p for p in paths if _is_test_path(p)]
        if test_paths:
            return _post_or_pre_patch(test_paths[0], has_patch, created_tests)
        return "L_navigate"

    if cls_name in TEST_RUN_LIST_FIELDS:
        values = getattr(action, TEST_RUN_LIST_FIELDS[cls_name], None) or []
        target = values[0] if values else None
        return _post_or_pre_patch(target, has_patch, created_tests, dynamic=not target)

    if cls_name in TEST_RUN_SCALAR_FIELDS:
        target = getattr(action, TEST_RUN_SCALAR_FIELDS[cls_name], None)
        return _post_or_pre_patch(target, has_patch, created_tests, dynamic=not target)

    if cls_name in TEST_RUN_DYNAMIC_CLASSES:
        return _post_or_pre_patch(None, has_patch, created_tests, dynamic=True)

    # Fallback for anything not explicitly mapped (e.g. `APIEndpoint`, which
    # is a nested schema element rather than a standalone registered action).
    return "general"


def _post_or_pre_patch(
    path: Optional[str],
    has_patch: bool,
    created_tests: Optional[Set[str]],
    *,
    dynamic: bool = False,
) -> str:
    if not has_patch:
        return "L_reproduce"
    if path and created_tests:
        for created_path in created_tests:
            if path == created_path or path.endswith(created_path) or created_path.endswith(path):
                return "V_newly_generated_test"
    if dynamic:
        return "V_newly_generated_test"
    return "V_regression_test"
