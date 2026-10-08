# Cadence: Strategic Guidance for Coding Agents

Cadence provides adaptive monitoring and intervention for coding agents. This repository implements Cadence and four comparison methods on mini-swe-agent and Moatless, with a shared SWE-bench execution and evaluation pipeline. Experiments run on Modal; local tools submit jobs and retrieve artifacts.

> **Acknowledgments.** We thank the authors of *Online Monitoring and Corrective Steering of Programming Agents* for their work; all baseline implementations in this repository were developed with reference to their [Agent-Planner repository](https://github.com/Intelligent-CAT-Lab/Agent-Planner).

## Installation

Python 3.10–3.13, Git, and Bash are required. Python 3.12 is recommended.

```bash
git clone --depth 1 https://github.com/LogAnalysisTech/Cadence.git
cd Cadence
bash scripts/setup.sh
source .venv/bin/activate
modal token new
```

The setup script installs local dependencies in `.venv`, creates `.env` if absent, and checks the CLI entry points. Agent dependencies are installed in the Modal images. To select a Python interpreter, use `PYTHON_BIN=/path/to/python3.12 bash scripts/setup.sh`.

## Configuration

Set credentials in the root `.env` file, following `.env.example`:

```dotenv
MODEL_API_ROUTE=openrouter
OPENROUTER_API_KEY=<your-api-key>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
VOYAGE_API_KEY=<your-voyage-key>
```

`VOYAGE_API_KEY` is required only for Moatless. Environment variables take precedence over `.env`; credentials are passed to Modal through Secrets. Modal authentication is managed separately by its local profile or `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` environment variables.

For direct agent access, set `MODEL_API_ROUTE=direct` and configure the corresponding `OPENAI_*` or `ANTHROPIC_*` variables in `.env`, using `.env.example` as the template. LLM-based monitors continue to use the OpenRouter credentials and endpoint.

The examples below use OpenRouter. Agent model names include the LiteLLM routing prefix, e.g. `openai/google/gemini-2.5-flash`; monitor names use the OpenRouter model ID directly, e.g. `openai/gpt-5.2`.

## Methods

Both agents use the following method selectors:

| Method | `--monitor-mode` | `--liveplan-variant` | Intervention |
|---|---|---|---|
| Vanilla | `vanilla` | Omit | No monitoring |
| RuleOnly | `liveplan` | `monitor` | Rule-based intervention |
| Fixed-Interval | `liveplan` | `refiner_only` | Periodic LLM advice; default interval 5 |
| Full-LivePlan | `liveplan` | `refiner` | Rule-triggered LLM refinement and plan updates |
| Cadence | `liveplan` | `adaptive_interval` | Adaptive scheduling and severity-dependent intervention |

`--liveplan-refiner-model` selects the monitor LLM and is unnecessary for Vanilla and RuleOnly. The CLI retains historical method identifiers for compatibility with recorded experiments.

## Experiments

### mini-swe-agent

Run Cadence on a single SWE-bench Lite instance:

```bash
modal run -m monitor_runtime.runners.mini_swe_agent \
  --run-id mini-cadence-example \
  --dataset lite --cases astropy__astropy-12907 \
  --agent-model openai/google/gemini-2.5-flash \
  --monitor-mode liveplan --liveplan-variant adaptive_interval \
  --liveplan-refiner-model openai/gpt-5.2 \
  --step-limit 150 --parallelism 2
```

Replace the method selectors using the table above and assign a distinct run ID to each experiment. Use `--parallelism 2` or higher with the current mini runner to produce the per-instance prediction files required by the evaluator.

Cadence scheduling ablations are available through `--adaptive-interval-fixed`, `--adaptive-interval-base`, and `--adaptive-interval-growth`. Intervention ablations use `--adaptive-interval-execution always-advisory` or `always-replacement`; the default is `severity`. These options are currently exposed only by the mini runner. Fixed-schedule Cadence retains Cadence's prompts and intervention logic and is distinct from the Fixed-Interval baseline.

### Moatless

Prepare the retrieval indices once for the selected instances, then run the experiment:

```bash
modal run -m monitor_runtime.runners.moatless \
  --dataset lite --cases astropy__astropy-12907 \
  --prepare-indices --parallelism 1

modal run -m monitor_runtime.runners.moatless \
  --run-id moatless-cadence-example \
  --dataset lite --cases astropy__astropy-12907 \
  --agent-model openai/google/gemini-2.5-flash \
  --monitor-mode liveplan --liveplan-variant adaptive_interval \
  --liveplan-refiner-model openai/gpt-5.2 \
  --parallelism 2
```

Indices are reused across methods. The runner uses `configs/moatless-swebench.json`, with a limit of 50 iterations and `max_cost=2.0` per instance. It does not expose `--step-limit`.

### Full benchmark

`--cases` specifies a comma-separated list of instance IDs. To select all 300 SWE-bench Lite instances:

```bash
CASES=$(python - <<'PY'
from datasets import load_dataset
ids = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")["instance_id"]
assert len(ids) == len(set(ids)) == 300
print(",".join(ids))
PY
)
```

Use `--cases "$CASES"` in the generation, index preparation, and evaluation commands. Both runners also support `--dataset verified`. Adjust `--parallelism` to the available service quotas; `--continuous-map` maintains a rolling set of active workers. Selecting a dataset alone does not select all its instances.

## Evaluation and artifacts

Evaluate generated patches using the same run ID, dataset, and instance list:

```bash
modal run -m monitor_runtime.evaluation.swebench \
  --run-id mini-cadence-example \
  --dataset lite --cases astropy__astropy-12907 --parallelism 2

modal run -m monitor_runtime.evaluation.packager \
  --run-id mini-cadence-example

mkdir -p downloads
modal volume get rafd-mini-swe-agent-artifacts \
  /archives/mini-cadence-example.tar.gz downloads/mini-cadence-example.tar.gz
```

Both agents write to `/<run-id>/` in the `rafd-mini-swe-agent-artifacts` Modal Volume. Evaluation results are recorded in `evaluation_modal/summary.json`; incomplete evaluations should be distinguished from unresolved instances. The packager includes existing run outputs and reports the archive's size and SHA-256 checksum.

Published results are available in [Cadence-Experiment-Results](https://huggingface.co/datasets/dengdan1999/Cadence-Experiment-Results). Corresponding download links and checksums are recorded in `coding_agent-baselines/<agent>/experiments/**/ARTIFACTS.txt`.

## Code organization

- `cadence/`: Cadence monitor, prompts, and agent adapters.
- `monitor_baseline/`: comparison methods and vendored dependencies; see the [provenance notice](monitor_baseline/vendor/NOTICE.md).
- `monitor_runtime/`: configuration, shared runtime, Modal runners, evaluation, and packaging.
- `coding_agent-baselines/`: agent implementations and experiment artifact references.

Additional CLI options are available through `modal run -m <module> --help` for each runner, evaluator, and packager above.
