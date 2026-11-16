# FraudFeat: Benchmarking and Automating Fraud-Graph Feature Mining with Self-Evolving Agents

This repository provides the **FraudFeat** benchmark, a separately released dataset, and the evaluation suite, together with **EvoFeat**, our proposed self-evolving multi-agent method for automated fraud-graph feature discovery.

The **Explorer** in EvoFeat discovers fraud patterns and evidence chains, the **Synthesizer** implements executable feature plans with validation and repair, and the **Evaluator** predicts feature utility and selects candidates for measured IV. Episode-wise feedback updates the feature pool, evaluator memory, and reusable playbooks. See the [EvoFeat guide](EvoFeat/README.md) for reproduction instructions.

FraudFeat pairs typed Parquet node and edge tables with an English data dictionary, account-level fraud labels, and a Python package for evaluating graph-derived features. Its evaluation package measures **effectiveness**, **structural diversity**, **semantic diversity**, and **interpretability**. Features can be submitted as DuckDB SQL queries or constrained Gremlin plans.

The released dataset is de-identified.

## Repository layout

```text
.
├── EvoFeat/                    # Proposed self-evolving agent method
│   ├── src/                    # Framework, experiments, and baselines
│   ├── config/                 # Model, execution, and graph configuration
│   ├── scripts/                # Reproduction entry points
│   └── examples/               # Illustrative playbook entries and feature plans
├── data/                       # FraudFeat dataset (Release asset; not versioned)
│   ├── schema.json
│   ├── labels.parquet
│   ├── splits/                 # Official fixed splits (shipped with the dataset)
│   │   ├── target_streams_s{42,43,44}.json
│   │   ├── offline_panel.parquet
│   │   └── evaluation_panel.parquet
│   ├── nodes/*.parquet
│   └── edges/rel_*.parquet
├── evaluation/                 # FraudFeat evaluation package
│   ├── fraudfeat/
│   ├── examples/               # SQL and Gremlin feature examples
│   ├── tests/
│   ├── pyproject.toml
│   └── README.md
├── LICENSE
└── README.md
```

## Data release

The dataset is distributed only through versioned GitHub Release assets; it is not stored in Git history. Code releases and data releases are versioned independently.

- Data release: [`data-v1.0.0`](../../releases/tag/data-v1.0.0)
- Archive: `fraudfeat-data-v1.0.0.tar.gz`
- Checksum manifest: `SHA256SUMS`

The archive contains one top-level `data/` directory. From a clone with the [GitHub CLI](https://cli.github.com/):

```bash
gh release download data-v1.0.0 \
  --pattern 'fraudfeat-data-v1.0.0.tar.gz' \
  --pattern 'SHA256SUMS'
sha256sum -c SHA256SUMS
tar -xzf fraudfeat-data-v1.0.0.tar.gz
```

On Windows, `tar -xzf fraudfeat-data-v1.0.0.tar.gz` also works in PowerShell; compare `Get-FileHash .\fraudfeat-data-v1.0.0.tar.gz -Algorithm SHA256` with `SHA256SUMS`. Alternatively, download both files from the release page, verify the checksum, and extract the archive at the repository root.

To keep the dataset elsewhere, supply its location through `--data` and `--schema`. No dataset files need to be committed.

## Dataset overview

| Component | Scale |
| --- | ---: |
| Entity types | 8 |
| Nodes | 12,394,721 |
| Relation types | 18 |
| Directed edges | 18,065,212 |
| Transaction edges | 5,319,577 across 10 relation types |
| Non-transaction edges | 12,745,635 across 8 relation types |
| Business attributes | 21 |
| Labeled `wechat_pay_account` nodes | 9,004,117 |
| Accounts marked as fraud (`label = 1`) | 276,702 |
| Accounts marked as non-fraud (`label = 0`) | 8,727,415 |

### Entity types

| Entity type (`entity_type`) | Nodes |
| --- | ---: |
| `wechat_pay_account` | 9,004,117 |
| `merchant_pay_account` | 536,580 |
| `individual_customer` | 1,191,147 |
| `natural_person` | 1,162,192 |
| `bank_card` | 366,090 |
| `phone_number` | 35,120 |
| `wechat_pay_bill_deletion_event` | 57,892 |
| `wechat_pay_card_back_information_query_event` | 41,583 |

### File format

- **Nodes:** `unique_id`, `entity_type`, and entity-specific attributes. `entity_type` values match `entities[].entity_type` in the schema.
- **Edges:** `src`, `dst`, and `relation`. Transaction edges also carry `total_amount`, `trade_count`, and `top_k`.
- **Labels:** `unique_id`, `label`; `1` marks an account as fraud and `0` marks it as non-fraud.
- **Schema:** `schema.json` is authoritative for entity types, fields, enum values, relation names, stored directions, coverage, `top_k`, missing values, and file locations.

### Official splits

The official splits are already prepared and shipped under `data/splits/`: `target_streams_s42.json`, `target_streams_s43.json`, `target_streams_s44.json`, `offline_panel.parquet`, and `evaluation_panel.parquet`. Use them as shipped; no split generation is required.

Transaction amounts use **RMB fen** (100 fen = 1 yuan). Relation identifiers follow the names in `relations[].en`; traversal follows the stored `src`/`dst` direction. Six personal/merchant transaction relations are bounded top-eight counterparty lists, while four bank-card transaction relations are per-account-and-card aggregates without top-eight truncation. Degrees from these six relations must therefore not be interpreted as total real-world counterparty degrees. See `coverage` and the relation-specific `top_k` definitions in the schema.

## Installation

Use **Python 3.9 or newer**. From the FraudFeat repository root, install the benchmark evaluation package first and then the EvoFeat method:

```bash
python -m pip install -e ./evaluation
python -m pip install -e ./EvoFeat
```

This installs the `fraudfeat` benchmark package and the `evofeat` method distribution with their command-line entry points. EvoFeat requires **`fraudfeat==1.0.0`**, provided by the local `evaluation/` package. Benchmark commands use the `fraudfeat-*` prefix; the method's main command is `evofeat`.

## Evaluation architecture

`evaluation/fraudfeat/` is the standalone, authoritative benchmark protocol for external SQL and constrained-Gremlin feature submissions. `EvoFeat/src/experiments/` contains the experiment harness: run-directory adapters, fixed-panel checks, episode curves, efficiency and cost metrics, and multi-seed aggregation.

Both layers use the same underlying IV, structural-diversity, semantic-diversity, and interpretability implementations. In particular, EvoFeat's Gremlin evaluation delegates to the benchmark's Gremlin engine, AutobinSplit, typed-graph/GED implementation, embedding protocol, and judge rubric/domain brief. SQL support is an additional standalone-submission interface, not a separate Gremlin metric definition.

For comparable reports, use the same fixed panel, feature-pool prefix, IV results, encoder/model, judges, and rubric fingerprint. EvoFeat-only episode trends, multi-judge analysis, efficiency, and cost reports extend the shared metrics rather than redefining them.

## EvoFeat quick start

After extracting FraudFeat under `data/`, run the complete EvoFeat pipeline from the repository root with Git Bash or WSL:

```bash
export DATA="/absolute/path/to/FraudFeat/data"
bash EvoFeat/scripts/run_public.sh "$DATA" evofeat_public_s42
```

The script loads the dataset's official splits (discovery targets and fixed panels), runs EvoFeat with seed 42, and evaluates the generated feature pool. Configure the model services described in the [EvoFeat guide](EvoFeat/README.md#model-configuration) first. Experiment outputs are written to `EvoFeat/runs/<run-id>/`.

## FraudFeat evaluation quick start

The commands below run from the repository root.

### 1. Use the official evaluation panel

The official final evaluation panel is provided at `data/splits/evaluation_panel.parquet`.

### 2. Evaluate effectiveness

DuckDB executes each SQL feature as a set-based query over the evaluation panel:

```bash
python -m fraudfeat.sql_effectiveness \
  --data data \
  --features evaluation/examples/features.sql.example.jsonl \
  --eval-panel data/splits/evaluation_panel.parquet \
  --out-dir results/effectiveness \
  --pool-limit 100
```

SQL features use the `evaluation_accounts`, `nodes`, and `edges` views and return `unique_id,value`. Feature values are scored with the shared `AutobinSplit` information-value estimator.

For constrained Gremlin feature plans, use `python -m fraudfeat.effectiveness` with `evaluation/examples/features.example.jsonl`. The same command accepts `EvoFeat/examples/features.json`; its [example guide](EvoFeat/examples/README.md#compute-real-iv) includes a complete IV command. For standard one-relation aggregates, use `python -m fraudfeat.direct_iv`.

### 3. Evaluate structural diversity

```bash
python -m fraudfeat.diversity \
  --features evaluation/examples/features.sql.example.jsonl \
  --schema data/schema.json \
  --iv-jsonl results/effectiveness/effectiveness.jsonl \
  --out-dir results/diversity \
  --pool-limit 100
```

Structural similarity combines anchored graph edit distance and statistic signatures. SQL submissions describe their structure through `graph_pattern`; Gremlin plans are parsed directly.

### 4. Evaluate semantic diversity and interpretability

Configure OpenAI-compatible model services:

```bash
export FRAUDFEAT_EMBED_BASE_URL="https://api.openai.com/v1"
export FRAUDFEAT_EMBED_API_KEY="<api-key>"
export FRAUDFEAT_EMBED_MODEL="text-embedding-3-small"
export FRAUDFEAT_JUDGE_BASE_URL="https://api.openai.com/v1"
export FRAUDFEAT_JUDGE_API_KEY="<api-key>"
export FRAUDFEAT_JUDGE_MODEL="gpt-4o-mini"
```

```bash
python -m fraudfeat.embedding_diversity \
  --features evaluation/examples/features.sql.example.jsonl \
  --iv-jsonl results/effectiveness/effectiveness.jsonl \
  --out-dir results/embedding \
  --pool-limit 100

python -m fraudfeat.interpretability \
  --features evaluation/examples/features.sql.example.jsonl \
  --schema data/schema.json \
  --iv-jsonl results/effectiveness/effectiveness.jsonl \
  --out-dir results/interpretability \
  --pool-limit 100
```

Semantic diversity reports pairwise cosine similarity and nearest-neighbor distance. Interpretability uses blind 1–5 scoring of feature meaning, computation logic, risk relevance, and actionability.

## Evaluation protocol and outputs

All four evaluators use the same feature-pool contract: `--pool-limit 100` selects the first 100 input entries, then IV ranks that pool for top-10/30/50/100 summaries. `--pool-limit 0` evaluates the full input. Use the same feature file, panel, and IV results across evaluators.

| Dimension | Main outputs |
| --- | --- |
| Effectiveness | `effectiveness.jsonl`, `effectiveness_summary.json`, `effectiveness_report.txt` |
| Structural diversity | `diversity_summary.json`, `diversity_report.txt` |
| Semantic diversity | `embedding_diversity.json`, `embedding_diversity_report.txt` |
| Interpretability | `judge_scores.jsonl`, `interpretability_summary.json`, `interpretability_report.txt` |

See [the evaluation guide](evaluation/README.md) for feature formats, Python APIs, local embedding models, judge configuration, and detailed metric definitions.

## Tests

After installation, run the FraudFeat evaluation test suite from the repository root:

```bash
python -m unittest discover -s evaluation/tests -t evaluation -v
```

The FraudFeat tests cover query execution, IV estimation, fixed-panel loading, feature-pool selection, structural similarity, and evaluator reports using synthetic fixtures.

## License

The dataset and accompanying documentation use [CC BY 4.0](LICENSE). The `fraudfeat` evaluation package uses Apache-2.0, as declared in [its package metadata](evaluation/pyproject.toml). The EvoFeat source code is released under the [MIT License](EvoFeat/LICENSE).
