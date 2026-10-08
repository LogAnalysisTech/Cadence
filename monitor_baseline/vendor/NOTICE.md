# NOTICE

The code under `monitor_baseline/vendor/` (`plan_monitor/`, `plan_refiner/`)
is vendored, with light stripping of scaffolding/tests/simulators, from:

- Upstream project: **Agent-Planner** ("LivePlan")
- Repository: https://github.com/Intelligent-CAT-Lab/Agent-Planner
- Pinned commit: `845b6a41c80a7c8c8b062ca04e7979d7f0b53228`

## License status

As of the pinned commit, the Agent-Planner repository root (where
`plan_monitor/` and `plan_refiner/` live) does **not** contain a LICENSE
file. Only the nested `SWE-agent/` and `SWE-bench_Pro-os/` submodules carry
their own licenses, and neither of those submodules is vendored here.

Because the license terms for the vendored code itself are unresolved:

- This code is for **internal research use only**.
- Do **not** include `monitor_baseline/vendor/` in any public code release
  or publication artifact for this project.
- Before any public repository release, `monitor_baseline/vendor/` must
  either be stripped from the release or the license question must be
  revisited with the upstream authors.

## Provenance

Files kept from `plan_monitor/`: `monitor.py`, `phases.py`, `rules.py`,
`commandParser.py`, `mapLang.py`, `buildGraph.py`, `config/*.json`.

Files kept from `plan_refiner/`: `refiner.py`, `formatters.py`, `models.py`,
`types.py`, `config/*.yaml`, `config/model_registry.json`, `utils/`.

The SWE-agent-specific `tool_configs` paths in
`plan_monitor/config/default_rules.json` were removed on 2026-10-08 when
SWE-agent support was retired. The paths previously pointed to byte-identical
copies of the upstream tool YAMLs. Mini-swe-agent uses shell-command parsing;
Moatless uses its structured-action role classifier.

The refiner model client also accepts `OPENROUTER_BASE_URL`; when unset, it
retains the upstream OpenRouter endpoint. This allows the shared runtime
configuration to select a compatible API gateway.
