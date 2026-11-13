# EvoFeat Illustrative Playbook and Feature Examples

This directory contains a small documentation sample of **48 illustrative skill entries** and **15 executable graph feature examples**. The skill entries demonstrate the published formats; they are not the full learned runtime state. Public plans use the public entity and relation identifiers, explicit stored traversal directions, and integer RMB fen for transaction amounts.

## Collection layout

```text
examples/
├── README.md
├── features.json
└── skills/
    ├── graph_paradigm.md
    ├── explore_navigate.md
    ├── feature_hypothesize.md
    ├── synthesis_playbook.md
    └── eval_calibration.md
```

The five playbooks use the framework's flat-file format. Static-Agent cold-start assets remain separate under `config/static_agent/`; the examples in this directory are not that method's frozen configuration.

## Skill collection

| Playbook | Entries | Coverage |
| --- | ---: | --- |
| [Graph paradigms](skills/graph_paradigm.md) | 10 | Customer and natural-person associations, two identity media, transfer-only topology, identity-transfer relays, merchant administration, withdrawals, and bank-card binding |
| [Navigation](skills/explore_navigate.md) | 7 | Identity-first exploration, relay completion, semantic pruning, multiple identity media, and merchant administration/receipt roles |
| [Feature hypotheses](skills/feature_hypothesize.md) | 10 | Counts, normalized reach, monetary aggregates, and existence tests |
| [Synthesis](skills/synthesis_playbook.md) | 10 | Schema grounding, stored direction, path coverage, seed anchoring, dependencies, supported query forms, ratio guards, reductions, deduplication, and monetary units |
| [Evaluator calibration](skills/eval_calibration.md) | 11 | Real predicted/measured IV pairs and reference-family matching by path order, endpoint type, transfer role, filters, depth, and statistic form |

Graph, navigation, hypothesis, and calibration entries are curated from appendix learning results; synthesis combines the relation-consistency lesson with reusable generated rules. The 48 entries are an illustrative subset, not a complete learned library or a measure of learning scale. Each published playbook contains only skill entries and their knowledge fields, without frontmatter, runtime learning counters, section headers, or changelogs. Navigation and hypothesis references resolve to the included graph paradigms.

## Feature collection

The collection covers distinct graph structures and statistic forms, including counts, normalized reach, monetary aggregates, and existence checks. Every feature carries self-contained `logic`, an ordered `evidence_chain` whose segments use only `from`, `rel`, `to`, `direction`, `hop`, and optional notes, and an explicit `expected_value_type`. Features use `feature_name` without a duplicate display-name field. The executable plans contain no stored IV results. The feature-value snippet and synthetic execution checks do not compute IV; use [Compute real IV](#compute-real-iv) below for panel-based IV.

| ID | Feature | Statistic |
| --- | --- | --- |
| F01 | Recipient reach through customer-linked upstream payers | Count |
| F02 | Identity-associated accounts of co-payers sharing a recipient | Count with identity qualification |
| F03 | Payment recipients of customer-associated accounts | Count |
| F04 | Reached customer records per initial customer record | Ratio |
| F05 | Natural-person reach per initial customer record | Ratio |
| F06 | Account expansion through two identity media | Ratio |
| F07 | Customer-record reach after a reverse transfer relay | Count |
| F08 | Natural-person reach after two forward payment steps | Count |
| F09 | Receipts of co-payers in a customer-linked neighborhood | Amount, fen |
| F10 | Withdrawals of customer-associated payment recipients | Amount, fen |
| F11 | Merchant payments after a reverse-relay neighborhood | Amount, fen |
| F12 | Receipts of merchants administered by associated accounts | Amount, fen |
| F13 | Merchant administration in an associated-account neighborhood | Existence |
| F14 | Four-step alternating transfer-account reach | Count |
| F15 | Bank-card binding after two customer bridges | Existence |

### F01–F03: Three different transfer neighborhoods

- **F01 — `customer_upstream_payer_recipient_count`:** expand the customer-associated account set, find its payers, then count the recipients of those payers. Initial accounts may reappear at the final layer.
- **F02 — `shared_recipient_customer_account_count`:** find accounts sharing one of the seed's recipients, expand their customer associations, and count reached accounts with a natural-person binding. A shared-recipient walk differs from two consecutive downstream payments.
- **F03 — `customer_associated_payment_recipient_count`:** count the direct payment recipients of customer-associated accounts. This compact identity-first path is distinct from F01's upstream/downstream fan-out and F02's transfer-first identity expansion.

### F04–F06: Normalization with explicit identity populations

- **F04 — `reached_customer_to_initial_customer_ratio`:** terminal customer-record count divided by initial customer-record count. Reached initial records remain in the numerator.
- **F05 — `natural_person_reach_per_customer_ratio`:** natural-person records reached through customer-associated accounts divided by the seed's initial customer-record count.
- **F06 — `alternating_identity_account_expansion_ratio`:** natural-person-first account expansion followed by customer-mediated account expansion, normalized by the initial natural-person record count. The denominator is an identity-record set, not an account set.

All three ratios use explicit count operands and return 0 for a zero denominator. Expansion ratios may exceed 1; they are not automatically proportions.

### F07–F08: Different identity endpoints after transfer paths

- **F07 — `customer_reverse_relay_terminal_customer_count`:** a customer-associated upstream/downstream transfer neighborhood ends at customer records.
- **F08 — `customer_forward_relay_natural_person_count`:** customer-associated accounts make a payment, the reached accounts make another payment, and the final recipient set contributes natural-person records. The second economic payment uses an incoming stored payee-side edge.

Changing the terminal entity or transfer direction changes the feature, even when both descriptions mention an identity-transfer relay.

### F09–F12: Four monetary objects

- **F09 — `customer_copayer_received_amount_fen`:** receipts of co-payers reached through a shared-recipient neighborhood. The final payee-side outgoing edges are the reached accounts' receipts, not their outgoing payments.
- **F10 — `customer_payment_relay_withdrawal_amount_fen`:** withdrawals of transfer recipients reached from customer-associated accounts. The amount belongs to the terminal withdrawal edges.
- **F11 — `customer_reverse_relay_receipt_amount_fen`:** receipt amounts on the fifth-hop payee-side edges of an upstream/downstream/upstream transfer walk rooted in customer-associated accounts.
- **F12 — `customer_administered_merchant_receipt_amount_fen`:** receipts of merchants administered by customer-associated accounts. Receipt payers may lie outside the initial identity neighborhood.

Each amount plan deduplicates contributing edges and retains **RMB fen**; 100 fen equals 1 yuan. Monetary scale is separate from the number of distinct endpoint accounts or instruments.

### F13–F15: Role, transfer-only comparison, and instrument association

- **F13 — `customer_associated_merchant_administration_flag`:** existence of a public super-administrator association in the customer-linked account neighborhood.
- **F14 — `alternating_transfer_four_step_account_count`:** a transfer-only out/in/out/in walk on the payee-side relation. It describes account topology rather than chronological movement of the same funds.
- **F15 — `two_customer_bridge_bank_card_binding_flag`:** a bank-card association after two customer-mediated account bridges. Binding is an association statistic, while F10 measures withdrawal activity.

## Execute feature values

Install the framework and the FraudFeat evaluation package as described in the [root guide](../README.md#installation). From the framework repository root, choose a released account ID:

```bash
export DATA="/absolute/path/to/fraudfeat-data"
export SEED_ID="<released-wechat-pay-account-id>"
```

The following evaluates the fifteen plans for that account and prints their values:

```python
import json
import os
from pathlib import Path

from system.execution.eg_plan import execute_plan
from system.execution.parquet_backend import ParquetExecutor

features = json.loads(Path("examples/features.json").read_text(encoding="utf-8"))
executor = ParquetExecutor(os.environ["DATA"])
for feature in features:
    value, trace, error = execute_plan(executor, feature["gremlin_plan"], os.environ["SEED_ID"])
    if error:
        raise RuntimeError(f"{feature['feature_name']}: {error}")
    print(feature["feature_name"], value)
```

This computes account feature values, not IV. The execution layer uses the framework's node-list limit, so the same limit should be retained when comparing plan values.

## Compute real IV

The standalone evaluator accepts the same 15 plans and computes their IV on the official 60,000-account panel. After installing the `evaluation` package, run:

```bash
python -m fraudfeat.effectiveness \
  --data "$DATA" \
  --features examples/features.json \
  --eval-panel "$DATA/splits/evaluation_panel.parquet" \
  --out-dir runs/published_example_effectiveness \
  --pool-limit 0 --workers 1
```

This writes per-feature IV, execution coverage, validity, errors, and a summary under `runs/published_example_effectiveness/`. The official discovery targets are excluded from final evaluation. `--pool-limit 0` keeps all 15 examples; one worker avoids duplicating the full in-memory graph, so increase it only when sufficient memory is available. For a smaller functional run, the offline panel can be used instead, but its IV values are online-selection diagnostics, not final reported metrics.

## Inspect structural diversity

Structural diversity can be inspected directly from the public dictionary:

```bash
python -m fraudfeat.diversity \
  --features examples/features.json \
  --schema "$DATA/schema.json" \
  --out-dir runs/example_structure \
  --pool-limit 0
```

## Validation

The published plans were validated before release on a small synthetic Parquet graph. The checks covered hand-specified count/ratio/amount/existence values, an empty seed, public relation names, both playbook directories, entry-only formatting, closed paradigm references, and reader/writer round trips. IV parity between EvoFeat and the standalone evaluator was also checked on a synthetic labeled panel. These synthetic checks do not replace real-data evaluation; real-data IV is available through the command above and the public evaluation workflow.

The standalone FraudFeat structural signature distinguishes all fifteen examples. The five books under `examples/skills/` remain independent of Static-Agent's cold-start books under `config/static_agent/`.
