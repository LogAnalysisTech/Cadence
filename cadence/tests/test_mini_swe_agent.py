from monitor_runtime.tests.mini_agent_fixtures import _FakeMiniAgent
from cadence.adapters.mini_swe_agent import MiniSWEAgentCadenceIntegration

import pytest
from cadence.monitor import CadenceMonitor


@pytest.mark.parametrize('mode', ['always-advisory', 'always-replacement'])
@pytest.mark.parametrize('verdict', ['advise', 'block', 'give_up'])
def test_mini_execution_ablation_controls_action_and_guidance_timing(mode, verdict):
    class Model:
        model_name = 'stub'
        @property
        def config(self):
            return self
        def query(self, *args):
            return f'<verdict>{verdict}</verdict><new_plan>Check the fix.</new_plan>', {}

    monitor = CadenceMonitor(refiner_model='stub', execution_mode=mode)
    monitor._model = Model()
    monitor._executed_step_count = 5
    eid = monitor.overhead.record(component='agent', operation='agent_generation', latency_seconds=0)
    message = {'role': 'assistant', 'content': 'proposed thought', 'extra': {'overhead_event_id': eid}}
    agent = _FakeMiniAgent([message])
    action = {'command': 'echo proposal'}
    integration = MiniSWEAgentCadenceIntegration(monitor)
    decision = integration.inspect_before_action(agent, message, action)
    assert monitor.refiner_interval == (7 if verdict == 'advise' else 5)
    assert not agent.env.executed_commands  # No forced submission, including give_up.
    if mode == 'always-advisory':
        assert decision.execute_proposed_step
        assert agent.messages == [message]  # Guidance is not injected before action.
        result = agent.env.execute(action)
        guidance = integration.handle_after_action(agent, message, action, result)
        assert 'Check the fix.' in guidance
        assert agent.env.executed_commands == ['echo proposal']
        assert monitor.executed_step_count == 6
    else:
        assert not decision.execute_proposed_step
        assert message not in agent.messages
        assert agent.messages[-1]['role'] == 'user'
        assert 'Check the fix.' in agent.messages[-1]['content']
        assert monitor.executed_step_count == 5
    event = next(e for e in monitor.overhead.payload()['events'] if e['event_id'] == eid)
    assert event['outcome'] == ('accepted' if mode == 'always-advisory' else 'discarded')
