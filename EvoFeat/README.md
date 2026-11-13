# EvoFeat

**EvoFeat** is the proposed self-evolving multi-agent method accompanying the [FraudFeat benchmark](../README.md). It discovers, implements, and evaluates graph-derived fraud-risk features on the FraudFeat dataset.

The **Explorer** discovers fraud patterns and evidence chains, the **Synthesizer** turns hypotheses into executable feature plans with validation and repair, and the **Evaluator** predicts feature utility and selects candidates for measured IV. Episode-wise feedback updates the feature pool, evaluator memory, and reusable playbooks.

## Repository layout

```text
.
├── src/
│   ├── system/                 # Multi-agent framework
│   │   ├── agents/             # Exploration, synthesis, and evaluation
│   │   ├── core/               # Graph model, schema, and vocabulary
│   │   ├── data/               # Input loaders and graph resources
│   │   ├── execution/          # Feature execution and Parquet backend
│   │   ├── evolution/          # IV feedback, memory, and playbook evolution
│   │   ├── llm/                # Model client and rate limiting
│   │   ├── memory/             # Retrieval implementation
│   │   └── orchestrator/       # Online/offline episodes and gating
│   ├── experiments/            # Preparation, baselines, and reporting
│   └── PE_Baseline/            # Manual, Stat, and Comm implementations
├── config/                     # Model, execution, IV, schema, and frozen playbooks
├── scripts/                    # Runnable experiment entry points
├── examples/                   # Learned playbook entries and executable feature plans
├── pyproject.toml
├── requirements.txt
└── README.md
```

Generated baseline snapshots and experiment resources are stored under EvoFeat's local `data/` directory. Experiment outputs are grouped under `runs/<run-id>/`. Official discovery targets and fixed panels come from the released dataset's `splits/` directory; no target or split generation is required. The FraudFeat graph itself is read from the dataset directory supplied with `--data`.

## Installation

Use **Python 3.9 or newer**. Create and activate a virtual environment from the **FraudFeat repository root**:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with `.\.venv\Scripts\Activate.ps1`. Bash scripts can be run through Git Bash or WSL; Python module commands work directly on Windows.

Install the sibling FraudFeat evaluation package and EvoFeat from the FraudFeat root:

```bash
python -m pip install -e ./evaluation
python -m pip install -e ./EvoFeat
```

EvoFeat depends on **`fraudfeat==1.0.0`**, provided by `evaluation/`. Download the dataset through the repository's [Data release](../README.md#data-release) and extract it into the repository-level `data/` directory by default; EvoFeat accepts another dataset location through `--data`.

## Dataset vocabulary

Entity types, node filenames, and relation endpoints use the same public identifiers:

| Entity type | Role |
| --- | --- |
| `wechat_pay_account` | Personal payment account |
| `merchant_pay_account` | Merchant payment account |
| `individual_customer` | Customer record |
| `natural_person` | Demographic and credential attributes |
| `bank_card` | Bank-card endpoint |
| `phone_number` | Phone-number endpoint |
| `wechat_pay_bill_deletion_event` | Bill-deletion event |
| `wechat_pay_card_back_information_query_event` | Card-back information query event |

The `natural_person` attributes are `birthday`, `cre_end_date`, `cre_start_date`, `cre_type`, and `gender`. The local [graph dictionary](config/fraudfeat_schema.json) mirrors the released `data/schema.json` for all dataset definitions and adds only the `version` and `format` metadata required by the EvoFeat runtime. Keep the two files synchronized when publishing a new data release.

## Playbooks

The Explorer uses three Markdown playbooks; the Synthesizer and Evaluator each use one. The five files share a single flat directory, both for live runs and for the frozen Static-Agent assets:

```text
skills/
├── graph_paradigm.md          # Explorer: graph structure paradigms
├── explore_navigate.md        # Explorer: navigation rules
├── feature_hypothesize.md     # Explorer: paradigm-to-feature rules
├── synthesis_playbook.md      # Synthesizer: plan-writing rules
└── eval_calibration.md        # Evaluator: family-level IV calibration rules
```

Unscoped live runs keep them in `data/memory/skills/`; with `--run-id`, they are isolated under `runs/<run-id>/memory/skills/`. Static-Agent reads the same five filenames directly from `config/static_agent/`, next to its `eval_memory.json`. At run time, each directory contains the five Markdown files; history and retired entries go into `_history/` and `_archive/` subdirectories.

## Published examples

The [example collection](examples/README.md) includes a small documentation sample of **48 illustrative skill entries** under `examples/skills/`: 10 graph paradigms, 7 navigation entries, 10 feature hypotheses, 10 synthesis rules, and 11 evaluator-calibration entries. It is not the full learned runtime state. These published entries deliberately omit runtime learning counters; live playbooks maintain their own statistics fields.

[examples/features.json](examples/features.json) provides **15 executable feature plans** covering identity records, transfer-first and identity-first neighborhoods, explicit normalization, merchant payments and administration, withdrawals, and bank-card binding. The plans are execution-checked against synthetic Parquet fixtures. They contain no stored IV results, but the standalone `fraudfeat.effectiveness` evaluator computes their real IV on the official panel.

Static-Agent cold-start assets remain separate under `config/static_agent/`. The example guide documents read-only feature-value execution, real IV computation, and structural-signature checks.

## Model configuration

Configure generation through environment variables or [config/llm.yaml](config/llm.yaml):

```bash
export SYSTEM_LLM_API_URL="<chat-completions-endpoint>"
export SYSTEM_LLM_API_KEY="<api-key>"
export SYSTEM_LLM_MODEL="<model-id>"
```

`SYSTEM_LLM_API_URL` is the complete chat-completions URL. Model sampling parameters, token limits, timeouts, request quotas, and concurrency are configured in `config/llm.yaml`.

| Purpose | Configuration |
| --- | --- |
| Generation | `SYSTEM_LLM_API_URL`, `SYSTEM_LLM_API_KEY`, `SYSTEM_LLM_MODEL` |
| Evaluator-memory embeddings | `SYSTEM_LLM_EMBEDDING_URL`, `SYSTEM_LLM_EMBEDDING_API_KEY`, `SYSTEM_LLM_EMBEDDING_MODEL` |
| Interpretability evaluation | `FRAUDFEAT_JUDGE_BASE_URL`, `FRAUDFEAT_JUDGE_API_KEY`, `FRAUDFEAT_JUDGE_MODEL` |
| Semantic-diversity evaluation | `FRAUDFEAT_EMBED_BASE_URL`, `FRAUDFEAT_EMBED_API_KEY`, `FRAUDFEAT_EMBED_MODEL` |

FraudFeat evaluation uses OpenAI-compatible base URLs, for example `https://api.openai.com/v1`. Use `--retriever-mode token` for token-overlap memory retrieval, or `hybrid` for embedding-assisted retrieval. The embedding and judge configurations are shared with the standalone FraudFeat evaluators.

## Quick start

Run the following commands from the `EvoFeat/` directory. Set `DATA` to the FraudFeat dataset directory containing `schema.json`, `nodes/`, `edges/`, `labels.parquet`, and the official `splits/` shipped alongside them:

```bash
export DATA="/absolute/path/to/FraudFeat/data"
bash scripts/run_public.sh "$DATA" evofeat_public_s42
```

This command loads the dataset's official splits, runs EvoFeat with seed 42, and evaluates the generated feature pool. Configure the generation, embedding, and judge services before running it.

The wrapper defaults to **25 episodes**, **4 target subgraphs per episode**, and **100 discovery targets**. Customize it through `SEED`, `EPISODES`, `EPISODE_SIZE`, `TARGET_SAMPLE`, and `PY`. Set `NO_EMBED=1` or `NO_JUDGE=1` to select which final evaluation dimensions to run.

### Official splits

The dataset ships fixed splits under `<DATA>/splits/`; no preparation or sampling step exists in this release:

| Artifact | Location |
| --- | --- |
| Discovery targets (3 streams × 100) | `<DATA>/splits/target_streams_s{42,43,44}.json` |
| Offline IV panel (1,000 label-1 + 5,000 label-0) | `<DATA>/splits/offline_panel.parquet` |
| Final evaluation panel (10,000 label-1 + 50,000 label-0) | `<DATA>/splits/evaluation_panel.parquet` |

Run one seed directly:

```bash
python -m system.main \
  --data "$DATA" \
  --offline-panel "$DATA/splits/offline_panel.parquet" \
  --targets "$DATA/splits/target_streams_s42.json" \
  --run-id ours__s42 --random-seed 42 \
  --target-sample 100 --max-hops 5 --max-episodes 25 --episode-size 4
```

The installed `evofeat` command exposes the same arguments as `python -m system.main`; `risk-features` remains available as a compatibility alias. Benchmark evaluation commands use the separate `fraudfeat-*` prefix.

## Methods and ablations

Each method script accepts the dataset path as its first argument. `SEEDS` defaults to `42 43 44`; for a single seed:

```bash
SEEDS="42" bash scripts/run_ours.sh "$DATA"
SEEDS="42" bash scripts/run_manual.sh "$DATA"
```

The experiment scripts retain `ours` in filenames and run IDs for compatibility with the original experiment artifacts; this method is **EvoFeat**.

| Method | Script | Setting |
| --- | --- | --- |
| EvoFeat (Ours) | `scripts/run_ours.sh` | Full self-evolving framework |
| Manual | `scripts/run_manual.sh` | Expert-prompted feature generation |
| Stat | `scripts/run_stat.sh` | Statistical feature generation |
| Comm | `scripts/run_comm.sh` | Community-oriented feature generation |
| Base-Agent | `scripts/run_base_agent.sh` | Agent pipeline with evolution and coverage guidance disabled |
| Static-Agent | `scripts/run_static_agent.sh` | Frozen playbooks and evaluator memory |
| w/o Explorer | `scripts/run_wo_exp.sh` | Random subgraph exploration |
| w/o Synthesizer | `scripts/run_wo_syn.sh` | Joint hypothesis/plan generation |
| w/o Evaluator | `scripts/run_wo_eval.sh` | Random offline gating |
| w/o Coverage | `scripts/run_wo_cov.sh` | Coverage guidance disabled |
| Evolution study | `scripts/run_ours_rq2.sh` | Full 25-episode run |

For Static-Agent, first build the frozen assets from the Manual, Stat, and Comm feature pools:

```bash
python -m experiments.static_agent_materialize \
  --baseline-run runs/manual__s42 \
  --baseline-run runs/stat__s42 \
  --baseline-run runs/comm__s42

SEEDS="42" bash scripts/run_static_agent.sh "$DATA"
```

Materialization samples 60 distinct measured baseline features and writes `eval_memory.json` plus the five playbook Markdown files directly under `config/static_agent/`.

## Evaluation and reports

Evaluate an existing run:

```bash
bash scripts/run_public_evaluation.sh "$DATA" runs/ours__s42
```

The framework calls the installed FraudFeat evaluators for effectiveness, structural diversity, semantic diversity, and interpretability. Input paths are resolved relative to the directory from which the script is invoked. Effectiveness results are cached using both the evaluation-panel and feature-definition fingerprints; use `FORCE=1 bash scripts/run_public_evaluation.sh "$DATA" runs/ours__s42` to force recomputation. The paper protocol defaults to `deepseek-v4-pro`, `gpt-6.1-sol`, and `claude-opus-5.5` as judges and `Qwen3-Embedding-0.6B` as the semantic encoder. Set `JUDGES` to a comma-separated model list, `FRAUDFEAT_EMBED_MODEL` to another encoder, and `WORKERS` to the judge concurrency when running a different protocol.

The `experiments/` layer owns run adaptation, checkpoints, episode curves, efficiency/cost metrics, and repeated-run aggregation. Base metric computation is delegated to `evaluation/fraudfeat/`: the same Gremlin engine and validity gates, AutobinSplit, structural GED/signatures, embedding text and cosine/NN protocol, and judge rubric/domain brief are used in both layers. Thus standalone Gremlin results and EvoFeat experiment results are directly comparable when the panel, pool, encoder, and judges are the same.

All evaluators select the first `POOL_LIMIT` features in file order, then rank that pool by IV for top-10/30/50/100 summaries. The default is 100; use `POOL_LIMIT=0` for full-pool evolution analysis. IV tiers are **low: IV < 0.2**, **mid: 0.2 ≤ IV < 0.5**, and **high: IV ≥ 0.5**. Transaction amounts remain in **RMB fen** across execution and evaluation.

Aggregate repeated runs of the same method:

```bash
python -m experiments.public_aggregate \
  runs/ours__s42 runs/ours__s43 runs/ours__s44 \
  --out runs/ours_aggregate
```

Aggregation reports the mean and sample standard deviation of each metric across the supplied runs.

### Run artifacts

| Artifact | Contents |
| --- | --- |
| `features.json` | Feature pool, logic, executable plans, explanations, and episode statistics |
| `candidates.jsonl` | Hypotheses, outcomes, repair counts, errors, and predicted/measured IV |
| `progress.json`, `run_summary.json` | Episode progress and run-level summaries |
| `memory/skills/` | The five evolving playbook Markdown files |
| `iv_values/` | Measured offline feature values and IV cache |
| `effectiveness_summary.json` | Final-panel effectiveness metrics |
| `diversity_summary.json` | Structural-diversity metrics |
| `embedding_diversity.json` | Semantic-diversity metrics and embeddings |
| `interpretability_summary.json` | Judge scores, coverage, and agreement |
| `run.log` | Execution log |

Use a distinct `--run-id` for each experiment. `--episode-resume` continues from the saved episode checkpoint.

