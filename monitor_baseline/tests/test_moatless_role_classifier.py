"""Unit tests for `monitor_baseline.moatless_role_classifier`.

Real moatless `ActionArguments` classes are not imported here: this
environment's `coding_agent-baselines/moatless` package fails to import
(litellm version mismatch, unrelated to this work: `moatless.completion.base`
imports `RateLimitError` from `litellm.files.main`, which does not exist in
the installed litellm build). `classify_moatless_action`/`is_ending_action`
only look at `type(action).__name__` and plain attribute access, so they are
tested here against lightweight stand-ins carrying the same class names and
field shapes as the real classes (verified by reading
`coding_agent-baselines/moatless/moatless/actions/*.py` directly).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitor_baseline.moatless_role_classifier import (  # noqa: E402
    classify_moatless_action,
    is_ending_action,
)


def make(_class_name: str, **fields):
    """Build a stand-in object with the given class name and attributes."""

    stub_cls = type(_class_name, (SimpleNamespace,), {})
    return stub_cls(**fields)


# (class_name, kwargs, has_patch, expected_role)
NAVIGATE_CASES = [
    ("GlobArgs", {"pattern": "*.py"}),
    ("ListFilesArgs", {"directory": "src"}),
    ("GrepToolArgs", {"pattern": "foo"}),
    ("ReadFilesArgs", {"glob_pattern": "*.py"}),
    ("ViewDiffArgs", {}),
    ("FindCodeSnippetArgs", {"code_snippet": "def foo(): pass"}),
    ("FindFunctionArgs", {"function_name": "foo"}),
    ("FindClassArgs", {"class_name": "Foo"}),
    ("SemanticSearchArgs", {"query": "find the bug"}),
    ("CleanupContextArgs", {"file_paths": ["a.py"]}),
    ("ListTasksArgs", {}),
]

GENERAL_CASES = [
    ("RespondArgs", {"message": "hello"}),
    ("ThinkArgs", {"thought": "hmm"}),
]

PATCH_CASES = [
    ("AddCodingTasksArgs", {}),
    ("CreateTasksArgs", {}),
    ("CreateJsonArtifactArgs", {}),
    ("FinishCodingTasksArgs", {}),
    ("DefineAPIArgs", {"title": "API"}),
    ("RemoveCodingTasksArgs", {}),
    ("UpdateTaskArgs", {}),
]

ENDING_CASES = ["FinishArgs", "VerifiedFinishArgs", "RejectArgs"]


@pytest.mark.parametrize(("class_name", "kwargs"), NAVIGATE_CASES)
def test_navigate_classes_are_l_navigate(class_name, kwargs):
    action = make(class_name, **kwargs)
    assert classify_moatless_action(action, has_patch=False, created_tests=set()) == "L_navigate"
    assert classify_moatless_action(action, has_patch=True, created_tests=set()) == "L_navigate"


@pytest.mark.parametrize(("class_name", "kwargs"), GENERAL_CASES)
def test_general_classes(class_name, kwargs):
    action = make(class_name, **kwargs)
    assert classify_moatless_action(action, has_patch=False, created_tests=set()) == "general"


@pytest.mark.parametrize(("class_name", "kwargs"), PATCH_CASES)
def test_patch_classes_are_p(class_name, kwargs):
    action = make(class_name, **kwargs)
    assert classify_moatless_action(action, has_patch=False, created_tests=set()) == "P"


@pytest.mark.parametrize("class_name", ENDING_CASES)
def test_ending_classes(class_name):
    action = make(class_name, finish_reason="done")
    assert is_ending_action(action) is True
    # Ending actions do not participate in the L/P/V role vocabulary.
    assert classify_moatless_action(action, has_patch=True, created_tests=set()) == "general"


def test_non_ending_action_is_not_ending():
    action = make("BashArgs", command="ls")
    assert is_ending_action(action) is False


@pytest.mark.parametrize(
    ("class_name", "field"),
    [
        ("AppendStringArgs", "path"),
        ("CreatePythonFileArgs", "path"),
        ("CreateFileArgs", "path"),
        ("InsertLinesArgs", "path"),
        ("StringReplaceArgs", "path"),
    ],
)
def test_path_bearing_edit_non_test_path_is_p(class_name, field):
    action = make(class_name, **{field: "src/module.py"})
    assert classify_moatless_action(action, has_patch=False, created_tests=set()) == "P"


@pytest.mark.parametrize(
    ("class_name", "field"),
    [
        ("AppendStringArgs", "path"),
        ("CreatePythonFileArgs", "path"),
        ("CreateFileArgs", "path"),
        ("InsertLinesArgs", "path"),
        ("StringReplaceArgs", "path"),
    ],
)
def test_path_bearing_edit_test_path_before_patch_is_l_reproduce(class_name, field):
    action = make(class_name, **{field: "tests/test_module.py"})
    assert classify_moatless_action(action, has_patch=False, created_tests=set()) == "L_reproduce"


def test_created_test_file_after_patch_is_newly_generated():
    created = set()
    create_action = make("CreateFileArgs", path="tests/test_new.py")
    # Creating a test file before the patch records its path as "created this run".
    assert classify_moatless_action(create_action, has_patch=False, created_tests=created) == "L_reproduce"
    assert "tests/test_new.py" in created

    # After a patch, re-touching that same recorded path is "newly generated", not regression.
    reread_action = make("ReadFileArgs", file_path="tests/test_new.py")
    assert classify_moatless_action(reread_action, has_patch=True, created_tests=created) == "V_newly_generated_test"


def test_regression_test_after_patch():
    action = make("RunTestsArgs", test_files=["tests/test_existing.py"])
    assert classify_moatless_action(action, has_patch=True, created_tests=set()) == "V_regression_test"


def test_run_tests_before_patch_is_l_reproduce():
    action = make("RunTestsArgs", test_files=["tests/test_existing.py"])
    assert classify_moatless_action(action, has_patch=False, created_tests=set()) == "L_reproduce"


def test_run_python_code_dynamic_validation():
    action = make("RunPythonCodeArgs", code="assert 1 == 1")
    assert classify_moatless_action(action, has_patch=False, created_tests=set()) == "L_reproduce"
    assert classify_moatless_action(action, has_patch=True, created_tests=set()) == "V_newly_generated_test"


def test_bash_non_test_command_is_general():
    action = make("BashArgs", command="ls -la")
    assert classify_moatless_action(action, has_patch=False, created_tests=set()) == "general"


def test_bash_pytest_command_is_test_related():
    action = make("BashArgs", command="pytest tests/test_module.py")
    assert classify_moatless_action(action, has_patch=False, created_tests=set()) == "L_reproduce"
    assert classify_moatless_action(action, has_patch=True, created_tests=set()) == "V_regression_test"


def test_edit_action_view_is_navigate():
    action = make("EditActionArguments", command="view", path="src/module.py")
    assert classify_moatless_action(action, has_patch=False, created_tests=set()) == "L_navigate"


def test_edit_action_str_replace_non_test_is_p():
    action = make("EditActionArguments", command="str_replace", path="src/module.py")
    assert classify_moatless_action(action, has_patch=False, created_tests=set()) == "P"


def test_edit_action_create_test_before_patch_is_l_reproduce():
    action = make("EditActionArguments", command="create", path="tests/test_new.py")
    assert classify_moatless_action(action, has_patch=False, created_tests=set()) == "L_reproduce"


def test_view_code_test_span_is_test_aware():
    span = SimpleNamespace(file_path="tests/test_module.py")
    action = make("ViewCodeArgs", files=[span])
    assert classify_moatless_action(action, has_patch=False, created_tests=set()) == "L_reproduce"
    assert classify_moatless_action(action, has_patch=True, created_tests=set()) == "V_regression_test"


def test_view_code_non_test_span_is_navigate():
    span = SimpleNamespace(file_path="src/module.py")
    action = make("ViewCodeArgs", files=[span])
    assert classify_moatless_action(action, has_patch=True, created_tests=set()) == "L_navigate"


def test_unmapped_class_falls_back_to_general():
    action = make("APIEndpoint", path="/users")
    assert classify_moatless_action(action, has_patch=False, created_tests=set()) == "general"
