from __future__ import annotations

import json
import os

import pytest

from monitor_runtime.experiment_config import ExperimentConfig, parse_ablations


@pytest.fixture(autouse=True)
def restore_monitor_environment():
    original = dict(os.environ)
    yield
    for key in list(os.environ):
        if key.startswith("REALTIME_MONITOR_") or key.startswith("MONITOR_BASELINE_") or key in {"MONITOR_EXPERIMENT_CONFIG", "PLAN_REFINER_RETRY_ATTEMPTS"}:
            os.environ.pop(key, None)
    for key, value in original.items():
        if key.startswith("REALTIME_MONITOR_") or key.startswith("MONITOR_BASELINE_") or key in {"MONITOR_EXPERIMENT_CONFIG", "PLAN_REFINER_RETRY_ATTEMPTS"}:
            os.environ[key] = value


def test_vanilla_clears_monitor_state_without_install_flags(monkeypatch) -> None:
    monkeypatch.setenv("REALTIME_MONITOR_SELECTOR_MODEL", "stale-selector")
    monkeypatch.setenv("MONITOR_BASELINE_ENABLED", "1")
    config = ExperimentConfig(monitor_mode="vanilla", agent_model="agent-model")

    config.apply_environment(agent_variant="mini-swe-agent", run_id="vanilla-run")

    assert os.environ["REALTIME_MONITOR_ENABLED"] == "0"
    assert os.environ["MONITOR_BASELINE_ENABLED"] == "0"
    assert "REALTIME_MONITOR_SELECTOR_MODEL" not in os.environ
    manifest = json.loads(os.environ["MONITOR_EXPERIMENT_CONFIG"])
    assert manifest["monitor_mode"] == "vanilla"
    assert manifest["agent_model"] == "agent-model"


def test_liveplan_has_its_own_variant_model_and_no_realtime_state(monkeypatch) -> None:
    config = ExperimentConfig(
        monitor_mode="liveplan",
        agent_model="agent",
        liveplan_variant="refiner",
        liveplan_refiner_model="anthropic/claude-sonnet-4.5",
        liveplan_refiner_config="config/default_with_rule.yaml",
    )

    config.apply_environment(agent_variant="moatless", run_id="baseline-run")

    assert os.environ["REALTIME_MONITOR_ENABLED"] == "0"
    assert os.environ["MONITOR_BASELINE_ENABLED"] == "1"
    assert os.environ["MONITOR_BASELINE_VARIANT"] == "refiner"
    assert os.environ["MONITOR_BASELINE_REFINER_MODEL"] == "anthropic/claude-sonnet-4.5"
    assert os.environ["PLAN_REFINER_RETRY_ATTEMPTS"] == "8"
    assert "REALTIME_MONITOR_SELECTOR_MODEL" not in os.environ


def test_liveplan_auto_refiner_config_leaves_variant_selection_to_factory(monkeypatch) -> None:
    config = ExperimentConfig(monitor_mode="liveplan", liveplan_variant="refiner_only")

    config.apply_environment(agent_variant="swe-agent", run_id="baseline-auto")

    assert "MONITOR_BASELINE_REFINER_CONFIG" not in os.environ


def test_adaptive_fixed_interval_payload_and_environment():
    from monitor_runtime.experiment_config import experiment_config_from_payload
    from monitor_runtime.factory import create_monitor
    config = ExperimentConfig(monitor_mode='liveplan', liveplan_variant='adaptive_interval', liveplan_refiner_model='openai/gpt-5.2', adaptive_interval_fixed=5)
    restored = experiment_config_from_payload(config.as_dict())
    assert restored == config
    restored.apply_environment(agent_variant='mini-swe-agent', run_id='fixed-ablation')
    monitor = create_monitor()
    assert monitor.fixed_interval == monitor.refiner_interval == 5
    assert monitor._model.config.model_name == 'openai/gpt-5.2'
    ExperimentConfig(monitor_mode='liveplan', liveplan_variant='adaptive_interval').apply_environment(agent_variant='mini-swe-agent', run_id='adaptive-default')
    assert create_monitor().fixed_interval == 0
    ExperimentConfig(monitor_mode='vanilla').apply_environment(agent_variant='mini-swe-agent', run_id='vanilla')
    assert 'MONITOR_BASELINE_ADAPTIVE_FIXED_INTERVAL' not in os.environ


@pytest.mark.parametrize('kwargs', [dict(adaptive_interval_fixed=-1), dict(adaptive_interval_fixed=5), dict(monitor_mode='liveplan', liveplan_variant='refiner_only', adaptive_interval_fixed=5)])
def test_reject_inapplicable_fixed_interval(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)


@pytest.mark.parametrize('base,growth', [(5, 4), (5, 6), (3, 2), (7, 2)])
def test_adaptive_hyperparameters_cli_payload_factory(base, growth):
    from monitor_runtime.runners.support import runner_experiment_config
    from monitor_runtime.experiment_config import experiment_config_from_payload
    from monitor_runtime.factory import create_monitor
    config = runner_experiment_config(agent_model='openai/google/gemini-2.5-flash', selector_model='gpt-5-mini', reviewer_model='openai/gpt-5.2', monitor_mode='liveplan', liveplan_variant='adaptive_interval', adaptive_interval_base=base, adaptive_interval_growth=growth)
    config = experiment_config_from_payload(config.as_dict())
    config.apply_environment(agent_variant='mini-swe-agent', run_id='robustness')
    monitor = create_monitor()
    assert (monitor.base_interval, monitor.growth_step, monitor.refiner_interval) == (base, growth, base)
    assert monitor.fixed_interval == 0
    ExperimentConfig(monitor_mode='vanilla').apply_environment(agent_variant='mini-swe-agent', run_id='cleanup')
    assert 'MONITOR_BASELINE_ADAPTIVE_BASE_INTERVAL' not in os.environ
    assert 'MONITOR_BASELINE_ADAPTIVE_GROWTH_STEP' not in os.environ


@pytest.mark.parametrize('kwargs', [{'adaptive_interval_base': 0}, {'adaptive_interval_base': 21}, {'adaptive_interval_growth': 0}, {'adaptive_interval_growth': True}, {'adaptive_interval_base': 3}, {'monitor_mode': 'liveplan', 'liveplan_variant': 'adaptive_interval', 'adaptive_interval_base': 3, 'adaptive_interval_fixed': 5}])
def test_invalid_adaptive_hyperparameter_config(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)


@pytest.mark.parametrize('mode', ['severity', 'always-advisory', 'always-replacement'])
def test_execution_ablation_cli_payload_factory_and_cleanup(mode):
    from monitor_runtime.runners.support import runner_experiment_config
    from monitor_runtime.experiment_config import experiment_config_from_payload
    from monitor_runtime.factory import create_monitor
    config = runner_experiment_config(monitor_mode='liveplan', agent_model='openai/google/gemini-2.5-flash', selector_model='unused', reviewer_model='openai/gpt-5.2', liveplan_variant='adaptive_interval', liveplan_refiner_model='openai/gpt-5.2', adaptive_interval_execution=mode)
    restored = experiment_config_from_payload(json.loads(json.dumps(config.as_dict())))
    assert restored == config
    restored.apply_environment(agent_variant='mini-swe-agent', run_id='ablation-test')
    monitor = create_monitor()
    assert monitor.execution_mode == mode
    assert monitor._model.config.model_name == 'openai/gpt-5.2'
    ExperimentConfig(monitor_mode='vanilla').apply_environment(agent_variant='mini-swe-agent', run_id='vanilla')
    assert 'MONITOR_BASELINE_ADAPTIVE_EXECUTION' not in os.environ


@pytest.mark.parametrize('kwargs', [
    {'adaptive_interval_execution': 'typo'},
    {'monitor_mode': 'vanilla', 'adaptive_interval_execution': 'always-advisory'},
    {'monitor_mode': 'liveplan', 'liveplan_variant': 'refiner_only', 'adaptive_interval_execution': 'always-replacement'},
])
def test_invalid_execution_ablation_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)


@pytest.mark.parametrize("mode", ["realtime", "monitor", "ours"])
def test_removed_monitor_modes_fail_before_configuration(mode):
    from monitor_runtime.runners.support import runner_experiment_config

    with pytest.raises(ValueError, match="legacy realtime monitor was removed"):
        runner_experiment_config(monitor_mode=mode, agent_model="agent", selector_model="unused", reviewer_model="unused")


@pytest.mark.parametrize("ablation", ["phase-policy", "finding-lifecycle", "capability-playbook"])
def test_removed_ablations_are_rejected(ablation):
    with pytest.raises(ValueError, match="Unknown ablation"):
        parse_ablations(ablation)
