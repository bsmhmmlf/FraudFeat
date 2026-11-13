# FraudFeat Evaluation

Standalone evaluation of graph-risk features on the released FraudFeat dataset:

- **Effectiveness:** batch DuckDB SQL for new features, direct one-relation aggregation, and a compatibility evaluator for existing constrained-Gremlin plans.
- **Structural diversity:** anchored graph edit distance (GED) and statistic signatures.
- **Semantic diversity:** pairwise cosine similarity and nearest-neighbor distance over embeddings.
- **Interpretability:** blind 1–5 scoring with an English rubric and a public-schema domain brief.

The package evaluates submitted features directly from Parquet and CSV files. Its installed Python API and command-line entry points can be used from any feature-generation project.

## 1. Installation and offline tests

Python 3.9 or later is required.

```bash
cd evaluation
python -m pip install -e .
python -m unittest discover -s tests -t . -v
```

Core dependencies are `numpy`, `pandas`, `pyarrow`, and `duckdb`. Remote judges and embeddings use only the Python standard library. Install the optional local encoder separately if needed:

```bash
python -m pip install -e ".[local-embed]"
```

## 2. Data and feature submissions

Download and verify the dataset through the repository's [Data release](../README.md#data-release). Extract it at the repository root to use the paths below, or pass another dataset location with `--data` and `--schema`.

The official splits are already prepared and shipped under `splits/`. Use them as shipped; no split generation is required:

```text
data/
├── schema.json
├── labels.parquet                 # Full source labels
├── splits/
│   ├── target_streams_s42.json    # Discovery targets (100 per stream, 3 streams)
│   ├── target_streams_s43.json
│   ├── target_streams_s44.json
│   ├── offline_panel.parquet      # 1,000 label-1 + 5,000 label-0
│   └── evaluation_panel.parquet   # 10,000 label-1 + 50,000 label-0
├── nodes/*.parquet
└── edges/rel_*.parquet
```

Each panel contains exactly `unique_id,label`. Pass `--eval-panel ../data/splits/evaluation_panel.parquet` to all three effectiveness evaluators. Do not combine `--eval-panel` with `--limit`, and do not replace the schema's full `labels.file` with a panel.

A feature submission can be a JSON array, a `{"features": [...]}` object, one JSON object, or JSONL. Each feature supports:

- `feature_name`
- `logic`
- `business_explanation` (optional)
- `expected_value_type`
- Either `sql` for DuckDB execution or `gremlin_plan` for compatibility execution
- `graph_pattern` (optional for SQL features; used by structural diversity)
- `pool_index` (optional stable identifier)
- `window` (optional legacy submission metadata; never shown to judges). EvoFeat run artifacts use `episode` for the same round information.

See `examples/features.sql.example.jsonl` for DuckDB SQL and `examples/features.example.jsonl` for constrained Gremlin.

### Analysis-pool contract

All four evaluators use the same rules:

1. `--pool-limit N` selects the **first N entries in input-file order**. The default is 100; `0` selects the full input.
2. IV is used only **after** this cutoff to rank entries within the selected pool for top-k metrics (`k=10/30/50/100`). It never pulls a high-IV entry from outside the prefix into the analysis pool.
3. An entry is identified by `feature_name#pool_index`. If `pool_index` is absent, its zero-based input position is used. Duplicate names therefore remain separate observations.
4. Missing IV values retain stable input order at the end of the ranking. Metric subsets with fewer than two entries report pairwise metrics as `null`.

For comparable reports, pass the same feature file, `--pool-limit`, and `effectiveness.jsonl` to every evaluator.

## 3. Effectiveness implementations

### 3.1 Batch DuckDB SQL for custom features

Use `sql_effectiveness.py` for new one-hop, multi-hop, filtered, or aggregate features. DuckDB reads the released tables directly; no graph database or table materialization is required:

```bash
python -m fraudfeat.sql_effectiveness \
  --data ../data \
  --features examples/features.sql.example.jsonl \
  --out-dir results/sql_effectiveness \
  --eval-panel ../data/splits/evaluation_panel.parquet \
  --pool-limit 100 \
  --threads 4
```

Each SQL feature executes once for the complete evaluation panel, not once per account. The following views are registered:

| View | Contents |
|---|---|
| `evaluation_accounts` | Label-free evaluation anchors; the only column is `unique_id` |
| `nodes` | All declared node tables, unioned by column name |
| `edges` | All declared relations with standardized `src`, `dst`, `relation`, and transaction fields |

No label table or label column is registered in DuckDB. Labels remain only in the Python scorer after SQL has returned feature values.

A query must be one read-only `SELECT` or `WITH` statement and return exactly two columns, `unique_id` and `value`, for every evaluation account exactly once. Start from `evaluation_accounts` and use `LEFT JOIN` so accounts without matching edges remain in the result. DDL/DML, multiple statements, and direct external-file functions are rejected. Query failures are isolated to the affected feature.

Raw `edges.total_amount` is in RMB fen. The additional `edges.total_amount_yuan` column equals `total_amount / 100.0`. SQL and Gremlin outputs use the same execution-coverage, nonzero-hit, nonconstant-value, AutobinSplit, and IV definitions. Output files retain the standard `effectiveness.jsonl`, `effectiveness_summary.json`, and `effectiveness_report.txt` names.

See `examples/features.sql.example.jsonl` for one-hop, two-hop, and amount-aggregation examples. Their `graph_pattern` fields provide structural metadata independently of SQL execution. SQL submissions should come from trusted authors; the query validator is not a general-purpose security sandbox.

### 3.2 Direct one-relation IV without Gremlin

Use `direct_iv.py` when the goal is to compute IV directly from the released tables. It does not read or translate `gremlin_plan`, and it does not import the Gremlin execution engine:

```bash
python -m fraudfeat.direct_iv \
  --data ../data \
  --relations personal_wechat_pay_transfer_payer_side,personal_wechat_pay_transfer_payee_side \
  --metrics degree,edge_count,trade_count_sum,amount_sum,amount_mean,amount_min,amount_max \
  --out-dir results/direct_iv \
  --eval-panel ../data/splits/evaluation_panel.parquet
```

The script reads `labels` and each selected public relation directly. It uses `join_fields.src/dst` only to determine which endpoints belong to the labeled account type. If both endpoints are `wechat_pay_account` nodes, it computes separate `src` and `dst` feature families. Available metrics are `degree`, `edge_count`, `trade_count_sum`, `amount_sum`, `amount_mean`, `amount_min`, `amount_max`, and `top_k_min`. Accounts without a matching relation receive value zero. The evaluation set must contain both label classes, and `--limit` preserves both classes when possible.

IV uses the same paper-aligned `AutobinSplit(smoothing_factor=0.5)` implementation. Constant features are invalid. Output files are `direct_iv.jsonl`, `direct_iv_summary.json`, and `direct_iv_report.txt`. Omit `--relations` to process every public relation. Use `--eval-panel` for a fixed panel, or omit it to sample with `--limit` or use all labels. Processing is serial by relation so the complete graph is not loaded into memory.

This direct path covers one-relation account aggregations. Use DuckDB SQL for custom multi-hop or combined aggregates, or the compatibility evaluator below for existing constrained-Gremlin files. Neither `direct_iv.py` nor `sql_effectiveness.py` imports the Gremlin execution engine.

### 3.3 Existing constrained-Gremlin plans

```bash
python -m fraudfeat.effectiveness \
  --data ../data \
  --features examples/features.example.jsonl \
  --out-dir results/effectiveness \
  --eval-panel ../data/splits/evaluation_panel.parquet \
  --pool-limit 100 \
  --workers 4
```

The evaluator validates each constrained-Gremlin plan, executes it for every evaluation account, and estimates IV with `AutobinSplit(smoothing_factor=0.5)`. The Laplace denominator is `N_y + |B_f| * epsilon`, where `|B_f|` is the actual bin count. Validity requires error-free execution, a nonzero hit, nonconstant numeric values, and computable IV; the evaluation set must contain both label classes. Semantic fidelity to the computation logic is scored separately by the interpretability evaluator.

All files declared by the schema are required. The loader builds an in-memory index of the complete graph; each additional worker creates another full index. Start with `--workers 1` on the full release and increase it only when memory permits.

Useful options:

| Option | Default | Meaning |
|---|---:|---|
| `--data` | `data` | Released data directory |
| `--features` | required | Feature JSON/JSONL file |
| `--out-dir` | `results` | Output directory |
| `--exclude-targets` | none | Discovery targets excluded from evaluation labels, not from traversal |
| `--eval-panel` | none | Fixed Parquet/CSV `unique_id,label` panel; mutually exclusive with `--limit` |
| `--limit` | `0` | Evaluation-account sample size; `0` means all |
| `--seed` | `0` | Evaluation-account sampling seed |
| `--unit` | `fen` | `fen`: no conversion and matches EvoFeat; `yuan`: threshold ×100 and aggregate ÷100 |
| `--workers` | `1` | Reader processes; each worker builds its own read-only index |
| `--pool-limit` | `100` | Input-prefix analysis-pool size; `0` means all |
| `--quiet` | false | Suppress the printed report |

Outputs:

- `effectiveness.jsonl`: one row per selected entry, including `pool_index`, `entry_key`, execution coverage, validity, IV, missing rate, and errors.
- `effectiveness_summary.json`: valid/invalid counts, median IV, IV≥0.2 and IV≥0.5 counts, top-k IV means, and leading entries.
- `effectiveness_report.txt`: human-readable report.

## 4. Structural diversity

```bash
python -m fraudfeat.diversity \
  --features examples/features.sql.example.jsonl \
  --schema ../data/schema.json \
  --iv-jsonl results/sql_effectiveness/effectiveness.jsonl \
  --out-dir results/diversity \
  --pool-limit 100 \
  --pairs
```

Gremlin plans are parsed directly. SQL features instead use their `graph_pattern` metadata; SQL text is not used to infer an authoritative graph structure. Both representations feed the same typed entity multigraph and statistic-signature implementation. The Gremlin parser supports `outE`, `inE`, `bothE`, `within` relation sets, branches, parallel edges, and endpoint-type compatibility checks. Entries without either structural representation are counted and skipped rather than collapsed into a spurious empty-structure cluster.

Similarity combines anchored GED (weight 0.75) and the statistic signature (weight 0.25). The signature compares operator, measured object, deduplication path, and expected value type with weights 0.3/0.3/0.2/0.2.

Outputs:

- `diversity_summary.json`: whole-pool and top-k structure counts, strict duplicate ratio, mean pairwise similarity, near-duplicate ratios at τ=0.70–0.95, and redundancy clusters.
- `diversity_report.txt`: human-readable report.
- `diversity_pairs.jsonl`: optional pairwise similarities for entries in the selected pool, keyed by stable entry identity.

## 5. Semantic diversity

### Offline pipeline check

```bash
python -m fraudfeat.embedding_diversity \
  --features examples/features.example.jsonl \
  --encoder hash \
  --out-dir results/embedding_hash \
  --pool-limit 100
```

The deterministic character-ngram `hash` encoder requires no network. It is a pipeline test only and must not be reported as semantic evidence.

### OpenAI-compatible embeddings

```bash
export FRAUDFEAT_EMBED_BASE_URL=https://api.openai.com/v1
export FRAUDFEAT_EMBED_MODEL=text-embedding-3-small
export FRAUDFEAT_EMBED_API_KEY=<your-api-key>

python -m fraudfeat.embedding_diversity \
  --features examples/features.example.jsonl \
  --encoder openai \
  --iv-jsonl results/effectiveness/effectiveness.jsonl \
  --out-dir results/embedding \
  --pool-limit 100
```

Use `--encoder sentence-transformers --model all-MiniLM-L6-v2` for a locally installed sentence-transformers model. `--probe` checks one embedding request without processing a feature file.

Embedding input is the versioned `Name/Logic/Query/Explanation` text. Remote response vectors are restored by response `index`, rounded to six decimals, and then used for cosine computation. Cache reuse requires matching encoder, model, input format, pool limit, pool size, ordered entry keys, feature-text SHA-1, and IV-ranking SHA-1. Use `--force` to recompute explicitly; `--no-vectors` omits raw vectors from the JSON artifact.

Outputs:

- `embedding_diversity.json`: metadata, pool entries, optional vectors, projection, pairwise matrix, and metrics.
- `embedding_diversity.txt`: machine-readable metric lines.
- `embedding_diversity_report.txt`: human-readable report.

## 6. Interpretability

All static judge prompts, the rubric, user-message labels, and the generated domain brief are English. Judges receive only feature name, computation logic, and optional business explanation—never labels, IV, method names, or episode/window numbers.

### Generate or freeze a domain brief

```bash
python -m fraudfeat.interpretability \
  --schema ../data/schema.json \
  --save-brief results/domain_brief.txt
```

The brief is generated solely from the supplied public schema and includes entity fields, enum dictionaries, public relation endpoints, and relation-specific `top_k` semantics. Edge counts are shown only when row-count metadata is actually present. A frozen `--brief` can be reused for an evaluation campaign.

### Fully offline mock run

```bash
python -m fraudfeat.interpretability \
  --features examples/features.example.jsonl \
  --schema ../data/schema.json \
  --judges mock \
  --out-dir results/interpretability_mock \
  --pool-limit 100
```

`mock` is deterministic and never opens a network connection. Its output is only a pipeline fixture, not an interpretability assessment.

### Remote judge

```bash
export FRAUDFEAT_JUDGE_BASE_URL=https://api.openai.com/v1
export FRAUDFEAT_JUDGE_MODEL=gpt-4o-mini
export FRAUDFEAT_JUDGE_API_KEY=<your-api-key>

python -m fraudfeat.interpretability \
  --features examples/features.example.jsonl \
  --brief results/domain_brief.txt \
  --iv-jsonl results/effectiveness/effectiveness.jsonl \
  --out-dir results/interpretability \
  --workers 4 \
  --stability-features 20 \
  --stability-repeat 3
```

Multiple judges can be supplied as a comma-separated `--judges` list. `--no-explanation` restricts the card to feature name and computation logic. `--probe` checks only the first judge.

Scoring rules:

- Only the first score (repeat 0) per entry and judge contributes to headline and top-k means.
- Additional repeats are used only for within-entry stability statistics.
- Each feature mean is the unweighted mean of its available judges' first scores.
- The overall mean weights each judge's first-score mean equally; it is not inflated by repeated rows.
- Scores outside the selected pool, ambiguous name-only scores for duplicate names, and scores from another rubric fingerprint are excluded from the current summary.

Outputs:

- `judge_requests.jsonl`: blind judge requests and rubric fingerprints.
- `judge_scores.jsonl`: scores with entry identity, judge, repeat, and rationale.
- `interpretability_summary.json`: coverage, first-score means, top-k means, judge agreement, Kendall's W, and repeat stability.
- `interpretability_report.txt`: human-readable report.

The rubric fingerprint covers the complete system message, and each request also carries a feature-card fingerprint. Changing feature text therefore invalidates its cached score automatically. After changing the rubric or domain brief, run with `--rescore`; scores from different prompts must not be mixed.

## 7. Python API

```python
from fraudfeat import GraphStore
from fraudfeat.effectiveness import evaluate_feature, load_eval_set, load_features

store = GraphStore("../data", load_labels=False)
eval_ids, labels = load_eval_set("../data")
feature = load_features("features.jsonl")[0]
row = evaluate_feature(store, feature, eval_ids, labels=labels)
print(row["entry_key"], row["iv"], row["valid"])
```

Lower-level execution is also available:

```python
from fraudfeat import execute_plan_local

value, trace, error = execute_plan_local(store, feature["gremlin_plan"], eval_ids[0])
```

## 8. Data semantics

- Nodes use `unique_id,entity_type` and entity-specific attributes such as `cre_type`. IDs are globally unique; `entity_type` follows `entities[].entity_type` in the schema, such as `wechat_pay_account` and `merchant_pay_account`.
- Edges use `src,dst,relation`; transaction edges add `total_amount,trade_count,top_k`. Relation names follow `relations[].en` and can be used in `has('relation', ...)` or edge-label steps. `join_fields` describes endpoint types, and the loader assigns in-memory edge identifiers for execution.
- Vertex properties use string comparison. Each time field uses the format declared for that field in `schema.json` (`yyyy-MM-dd` or `yyyy-MM-dd HH:mm:ss`). Missing-value sentinels are listed in `missing_value_policy`; unparsable values retain their original representation.
- Amounts use RMB fen (100 fen = 1 yuan). Enum meanings and text-valued merchant fields are described in English. For `enum_kind = text`, `values` lists all categories observed in this release; open-valued fields use their declared patterns instead. Forward and reverse traversal are interpreted relative to the stored `src`/`dst` direction.
- Six personal/merchant transaction relations are bounded top-eight counterparty lists, while four bank-card transaction relations are per-account-and-card aggregates without top-eight truncation. Degrees from these six relations are truncated by construction and are not total real-world counterparty degrees. Consult `coverage` and each relation's `top_k` definition.
- Labels use `unique_id,label`: `1` marks an account as fraud, and `0` marks it as non-fraud.
- `schema.json` is authoritative for entity and relation interpretations, attributes, value mappings, coverage, and file locations. Model endpoints and API keys are configured through the `FRAUDFEAT_JUDGE_*` and `FRAUDFEAT_EMBED_*` environment variables.
