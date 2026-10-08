"""Sanity check that the vendored `StatefulPhaseMonitor` and its tool-config
path rewrite (`monitor_baseline/vendor/plan_monitor/config/default_rules.json`)
load without error, and classify hand-written swe-agent-style bash commands
into the expected L/P/V roles.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import monitor_baseline.bridge  # noqa: E402  (inserts vendor/ onto sys.path)
from plan_monitor.monitor import StatefulPhaseMonitor  # noqa: E402
from plan_monitor.phases import ActionEvent  # noqa: E402


def test_monitor_constructs_and_loads_tool_configs():
    monitor = StatefulPhaseMonitor()
    assert monitor.rule_engine is not None
    assert monitor.parser is not None


def test_navigate_then_patch_then_regression_test_roles():
    monitor = StatefulPhaseMonitor()

    monitor.check_step_pre_emptively(
        ActionEvent(step_index=1, command="grep -rn 'foo' src/"), thought="looking around"
    )
    assert str(monitor.get_current_phase()) == "L_navigate"

    monitor.check_step_pre_emptively(
        ActionEvent(step_index=2, command="sed -i 's/foo/bar/' src/module.py"), thought="patching"
    )
    assert str(monitor.get_current_phase()) == "P"

    monitor.check_step_pre_emptively(
        ActionEvent(step_index=3, command="pytest tests/test_module.py"), thought="verifying"
    )
    assert str(monitor.get_current_phase()) == "V_regression_test"


def test_reproduce_command_before_patch_is_l_reproduce():
    monitor = StatefulPhaseMonitor()
    monitor.check_step_pre_emptively(
        ActionEvent(step_index=1, command="python tests/repro.py"), thought="reproducing the bug"
    )
    assert str(monitor.get_current_phase()) == "L_reproduce"
