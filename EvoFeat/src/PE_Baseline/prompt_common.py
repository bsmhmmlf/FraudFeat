import json
from datetime import date, datetime
from decimal import Decimal


GREMLIN_OUTPUT_CONTRACT = r"""
## Gremlin plan DSL

EasyGraph stores every vertex with label `entity` and every edge with label `relation`. The business relation is stored in the edge property `join_field`.
- Start every traversal with `g.V()` and anchor it with `{{SEED}}` or `{{STEP:sX}}`. Never perform an unanchored full-graph scan.
- Use `<business_vid>|entity` as a vertex token. The executor automatically restores this token after an intermediate `.id()` step.
- Traverse stored direction with `.outE().has('join_field', '<jf>').inV()` and reverse direction with `.inE().has('join_field', '<jf>').outV()`. Use `.bothE().has('join_field', '<jf>').otherV()` only when direction is genuinely irrelevant.
- For `xxx|payer`, src is the payer and dst is the payee. For `xxx|payee`, src is the payee and dst is the payer. For `fieldA&fieldB|EntityA&EntityB`, EntityA is src and EntityB is dst. `withdraw`, `charge`, and `from_card` relations run from wechat_pay_account to bank_card; stored direction is not necessarily fund-flow direction.
- Vertex properties are strings: write `has('state', 'Deregistered')`, not an unquoted value. Timestamp comparisons also use quoted strings.
- Edge `total_amount` and `trade_count` are integers. `total_amount` uses fen without rescaling.
- Supported Gremlin reductions are `sum()`, `mean()`, `count()`, `min()`, and `max()`. Do not use `avg()`, `asDouble()`, or `asLong()` in a traversal.

The `plan` object has this form:
{
  "steps": [
    {"id": "s1", "desc": "business role of this step", "depends_on": [], "output": "id_list",
     "gremlin": "g.V().hasId({{SEED}}).outE().has('join_field','<join_field>').inV().dedup().id().toList()"}
  ],
  "final": {"type": "count", "step": "s1"}
}

`{{SEED}}` is the target-account anchor. `{{STEP:sX}}` is the ID list returned by an upstream step; consume it as `g.V().hasId(within({{STEP:s1}}))` and list `s1` in `depends_on`. Placeholders already have string semantics and must not be quoted.
- `output="id_list"`: end with `.dedup().id().toList()`.
- `output="scalar"`: end with `sum()`, `count()`, `mean()`, `min()`, or `max()`.
- `output="scalar_list"`: end with `.toList()` for a Python-side reduction.
- Final forms are `{"type":"scalar"|"count"|"exists","step":"sX"}`, `{"type":"reduce","step":"sX","op":"sum|mean|min|max"}`, and `{"type":"ratio","numerator_step":"...","numerator_mode":"scalar|count","denominator_step":"...","denominator_mode":"scalar|count"}`.
- Compute ratios in Python from separate numerator and denominator steps, never by division inside Gremlin. A zero denominator produces zero deterministically.
- Use one hop per step, deduplicate intermediate ID lists, and make the plan implement every hop and the exact aggregation object stated in `logic`.
- Filters must come from `logic` or observed subgraph facts. Never invent thresholds or property filters.
- Inside `union()`, `where()`, or `local()`, use anonymous traversals such as `__.V()`, never `g.V()`. Calls to `.or()`, `.and()`, and `.not()` must contain anonymous traversals.
- Step IDs start with `s` and contain only letters, digits, and underscores.
- Never emit raw identity-document numbers, phone numbers, bank-card numbers, or real vertex IDs.

Return exactly one valid JSON object, with no Markdown fence or surrounding prose:
{
  "features": [
    {
      "name": "Concise English business-indicator name without hop counts, path notation, anonymous node IDs, or display-only grouping terms",
      "logic": "Self-contained English computation logic anchored at the target account, using exact entity, join_field, and property identifiers from the subgraph and ending in one scalar",
      "hops": 3,
      "predicted_iv": 0.35,
      "value_range": "non-negative integer, [0,1], or another precise scalar range",
      "rationale": "Two to four sentences explaining the business behavior, risk relevance, and possible strategy use",
      "plan": {"steps": [{"id": "s1", "desc": "business role", "depends_on": [], "output": "id_list", "gremlin": "g.V().hasId({{SEED}}).outE().has('join_field','<join_field>').inV().dedup().id().toList()"}], "final": {"type": "count", "step": "s1"}}
    }
  ],
  "feature_combinations": "Optional combination advice or an empty string",
  "limitations": "Optional limitations or an empty string"
}

Every feature must contain `plan` and `rationale`. `hops` must be an integer from 3 through 5 and must not exceed the available subgraph depth. `predicted_iv` is an honest estimate of classic unbounded WoE information value: below 0.02 is unpredictive, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, and above 0.5 very strong. The operational tiers are low below 0.2, medium from 0.2 through below 0.5, and high at 0.5 or above. Inflating every estimate only wastes the measurement budget.
If no defensible feature exists, return an empty `features` array rather than padding the output.
"""


BASE_FEATURE_SYSTEM_PROMPT = r"""You are a senior fraud and anti-money-laundering modeling expert. Discover concealed, subgraph-specific anomalies and turn them into executable scalar features.

The input is a directed multigraph whose vertices share the physical label `entity` and whose edges share the physical label `relation`. The public entity types are `wechat_pay_account`, `merchant_pay_account`, `individual_customer`, `natural_person`, `bank_card`, `phone_number`, `wechat_pay_bill_deletion_event`, and `wechat_pay_card_back_information_query_event`. Shared identifiers and transactions are already materialized as edges or shared intermediate vertices; traverse them instead of comparing raw node identifiers. No other entity, field, or relation is available.

Every feature must:
- be anchored at the target account and return one scalar for that account;
- use exact snake_case entity types, `join_field` values, and property names observed in the input;
- use 3, 4, or 5 hops, remain within the supplied subgraph, and be representable by a bounded lightweight plan;
- be self-contained, independently computable, and free of references to other output features;
- capture a concrete risk behavior rather than a generic topology label;
- explain why an ordinary account would be unlikely to attain a similar value.

Produce 3-4 high-quality features. At least two must combine identity-sharing relations with transaction or other differentiated relations within one path. No more than one third may rely solely on the target's subject. Cover multiple observed relation dimensions and at least three arithmetic paradigms: distinct counting, ratios, shortest distance or reachability, edge-strength aggregation, grouped or directional comparison, and multidimensional intersection. Do not emit isomorphic variants that merely change an entity type or hop count. Prefer semantically rich ratios, paths, intersections, and strength measures over brute-force breadth.

Feature names must be concise English business indicator names. Do not include hop counts, arrows, anonymous short IDs, path shapes, or display-only terms such as relation group or node summary. The `logic` and `plan` must use graph identifiers verbatim and must never use anonymous short IDs as executable identifiers.

The displayed target account, node summary, relations, member attributes, and mappings are input views, not additional graph entity types. Arrows used to organize a displayed path do not override the stored edge direction. Use relation roles and stored src/dst direction consistently.

Base every claim on the supplied subgraph. Do not invent external facts, unavailable labels, global statistics, or exact identifier-value filters. If a statement would remain equally true for an arbitrary risky-account subgraph, replace it with subgraph-specific evidence and computation.
"""


CONTEXTUAL_FEATURE_SYSTEM_PROMPT = BASE_FEATURE_SYSTEM_PROMPT + r"""
## Temporal causality

The target account has a fraud label. Use available registration, authentication, contact-update, and other vertex timestamps to state temporal direction in the analysis. Do not assume transaction timestamps because public trade edges do not contain them. Temporal reasoning guides feature selection and explanation; do not invent a temporal filter that the plan cannot support.

Useful perspectives include community bridging, independent-source convergence followed by downstream dispersion, multidimensional identity sharing, coordinated activity, and unusual bounded topology. These are perspectives, not a checklist or acceptable generic feature names.
"""


MANUAL_FEATURE_SYSTEM_PROMPT = BASE_FEATURE_SYSTEM_PROMPT + r"""
## Orthogonal structural operators

In addition to at least three arithmetic paradigms, cover at least three structural operators and avoid repeating the same structural-operator/arithmetic-paradigm pair:
- S1 outward expansion or inward convergence: reachable entities or independent sources arriving at the target.
- S2 neighborhood closure: bounded triangles, cycles, or interconnection among the target's neighbors.
- S3 common-neighbor overlap: shared intermediate entities between the target and another account.
- S4 multi-segment meta-paths: at least two heterogeneous edge types, including layered expansion ratios as a special case.

At least two features must use S4 meta-paths. Do not let every feature collapse into S1 outward counting. Global iterative centrality, global community detection, and time-series or graph-evolution features are excluded because they are not bounded lightweight target-anchored computations or are unsupported by the data. Do not introduce time-window constraints.

High-value features commonly combine a multi-segment meta-path with a role or edge-strength constraint and a scalar aggregation. Any role constraint must correspond to an entity or label actually present in the subgraph.
"""


MANUAL_USER_PROMPT_TEMPLATE = r"""Analyze the de-identified knowledge-graph subgraph centered on target account $target_id. The account is labeled risky, but do not merely restate that label. Find concealed structures that require multi-hop traversal or cross-relation composition and define executable features.

The actual maximum available depth is $max_hop. No node or edge beyond that depth exists in the input. Every feature must satisfy `3 <= hops <= min(5, $max_hop)`; if this is impossible, return no feature. At least half of the features should use the deepest defensible structure rather than stopping at a shallow count.

English entity, join_field, and property identifiers are used for computation. Use the supplied views only to understand business semantics. Write all natural-language output in English. `logic` must quote graph identifiers exactly, must be fully self-contained, and must explain the expected ordinary-account contrast and the risk meaning of a large value.

Do not copy examples or manufacture generic patterns. Cover at least three arithmetic paradigms and at least three structural operators, with no repeated structural/paradigm pair. Do not use global centrality, global community detection, or temporal-evolution features.

Subgraph data:
```json
$subgraph_json
```
"""


def prompt_data(value):
    if isinstance(value, dict):
        return {str(k): prompt_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [prompt_data(v) for v in value]
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _json_block(value) -> str:
    return json.dumps(prompt_data(value), ensure_ascii=False, indent=2)


def build_stat_round1_prompt(subgraph, graph_stats) -> str:
    return f"""You are a financial risk graph analyst. Analyze the target-centered subgraph using the deterministic graph statistics and raw subgraph below.

Graph statistics:
```json
{_json_block(graph_stats)}
```

Subgraph:
```json
{_json_block(subgraph)}
```

Address degree distribution, clustering, betweenness, k-core membership, transaction direction and concentration, and the most important preliminary risk signals. Distinguish observation from inference and cite concrete nodes or relations where available.

Return exactly one JSON object with this schema:
{{
  "round": 1,
  "degree_analysis": {{"target_degree_position": "high|medium|low", "abnormal_degree_nodes": [], "degree_distribution_skewness": "description"}},
  "clustering_analysis": {{"target_vs_avg": "above|below|near", "interpretation": "interpretation", "abnormal_nodes": []}},
  "betweenness_analysis": {{"target_is_bridge": true, "bridge_nodes": [], "connected_communities": "description"}},
  "kcore_analysis": {{"target_core_layer": 0, "max_core_layer": 0, "core_risk_nodes": [], "interpretation": "interpretation"}},
  "transaction_analysis": {{"has_transaction": true, "fund_direction": "net_inflow|net_outflow|bidirectional|none", "fast_in_fast_out": false, "counterparty_concentration": "concentrated|dispersed|none", "amount_pattern": "description", "risk_counterparties": []}},
  "risk_signals": {{"likely_pattern": "P1|P2|P3|P4|P5|P6", "key_suspicious_nodes": [], "abnormal_structures": [], "next_round_focus": "description"}},
  "analysis": "concise synthesis"
}}
"""


def build_stat_round2_prompt(subgraph, graph_stats, round1_response) -> str:
    return f"""You are a financial risk graph analyst. Use the Round 1 findings to perform a focused second-pass investigation.

Graph statistics:
```json
{_json_block(graph_stats)}
```

Round 1 response:
```json
{round1_response}
```

Subgraph:
```json
{_json_block(subgraph)}
```

Select the two or three most relevant directions: community bridging, independent-source convergence and downstream dispersion, dense-group structure, paths to risky nodes, multidimensional identity disguise, deep cross-hop paths, or transaction-funnel behavior. Trace concrete paths and relation fields, inspect upstream/downstream independence, and identify which observations would matter for an executable feature.

Return exactly one JSON object with this schema:
{{
  "round": 2,
  "focus_direction": "selected directions and rationale",
  "deep_dive_results": {{"key_findings": [], "suspicious_nodes": [{{"id": "...", "reason": "...", "attributes": "..."}}], "risk_paths": [{{"path": "A->B->C", "fields": [], "risk_level": "high|medium|low"}}]}},
  "structural_analysis": {{"community_structure": "description", "bridge_nodes": [], "dense_subgraphs": [], "anomalous_patterns": []}},
  "transaction_analysis": {{"fund_flow_summary": "description", "funnel_pattern": "description", "fast_in_fast_out": false, "amount_anomaly": "description", "tx_paths_to_risk": [{{"path": "A->B->C", "fields": [], "risk_level": "high|medium|low"}}]}},
  "analysis": "top three risk signals"
}}
"""


def build_stat_final_user_prompt(subgraph, graph_stats, multi_round_context) -> str:
    labels = ["statistical interpretation and risk signals", "focused deep dive"]
    sections = []
    for i, answer in enumerate(multi_round_context, 1):
        label = labels[i - 1] if i <= len(labels) else f"additional round {i}"
        sections.append(f"### Round {i}: {label}\n{answer}")
    context = "\n\n".join(sections)
    target = prompt_data(subgraph).get("target_account", {})
    target_id = target.get("id", "unknown") if isinstance(target, dict) else "unknown"
    return f"""Define executable risk features for the subgraph centered on target account {target_id}. The account is labeled risky, but do not merely restate the label. Discover concealed multi-hop or cross-relation structures.

Earlier analysis rounds are evidence, not a substitute for independently checking the raw subgraph:
{context}

Hard requirements: 3-4 features; hops in 3-5; at least two features combine identity-sharing and differentiated relations in one path; at least three arithmetic paradigms; no isomorphic hop/entity variants; concise English business names; exact graph identifiers in `logic`; and one bounded anchored scalar plan per feature. Prioritize structures that precede the risk label when temporal evidence is available.

Graph statistics:
```json
{_json_block(graph_stats)}
```

Raw subgraph:
```json
{_json_block(subgraph)}
```
""" + GREMLIN_OUTPUT_CONTRACT


def build_comm_l1_prompt(comm_subgraph, target_id, profile) -> str:
    return f"""You are a financial risk graph analyst. Summarize one community slice of the ego network around target account {target_id}; the target is included in the slice.

Deterministic community profile:
```json
{_json_block(profile)}
```

Community subgraph:
```json
{_json_block(comm_subgraph)}
```

Identify dominant relation fields, bounded topology, presence and distance of risky-label nodes, the most suspicious concrete observation, and key nodes. Return exactly one JSON object without Markdown:
{{
  "community_id": {profile['id']},
  "dominant_relation_field": "dominant field",
  "topology": "description",
  "has_black_label": true,
  "min_hop_to_target": {profile['min_hop_to_target']},
  "risk_point": "one or two sentences",
  "key_nodes": [],
  "summary": "concise English summary"
}}
"""


def build_comm_l2_prompt(super_id, l1_summaries, comm_connections) -> str:
    return f"""You are a financial risk graph analyst. Synthesize the L1 community summaries and inter-community connections for one super-community.

L1 summaries:
```json
{_json_block(l1_summaries)}
```

Inter-community connections:
{comm_connections}

Identify shared relation fields, possible cross-community risk paths, the most suspicious subcommunities, and the closest P1-P6 pattern tendency without treating that label as a conclusion. Return exactly one JSON object without Markdown:
{{
  "super_community_id": {super_id},
  "shared_relation_fields": [],
  "risk_transmission_chain": "description",
  "most_suspicious_subcommunities": [],
  "pattern_tendency": "P1|P2|P3|P4|P5|P6",
  "summary": "concise English synthesis"
}}
"""


def build_comm_l3_user_prompt(subgraph, target_id, l2_summaries, l1_summaries) -> str:
    return f"""Define executable risk features for the subgraph centered on target account {target_id}. The account is labeled risky, but do not merely restate the label. Discover concealed multi-hop or cross-relation structures.

The hierarchical community summaries below provide context but do not replace independent inspection of the raw subgraph.

L2 super-community summaries:
```json
{_json_block(l2_summaries)}
```

L1 community summaries:
```json
{_json_block(l1_summaries)}
```

Hard requirements: 3-4 features; hops in 3-5; at least two features combine identity-sharing and differentiated relations in one path; at least three arithmetic paradigms; no isomorphic hop/entity variants; concise English business names; exact graph identifiers in `logic`; and one bounded anchored scalar plan per feature. Prioritize structures that precede the risk label when temporal evidence is available.

Raw subgraph:
```json
{_json_block(subgraph)}
```
""" + GREMLIN_OUTPUT_CONTRACT
