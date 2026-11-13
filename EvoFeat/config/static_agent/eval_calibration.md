### multi_hop_entity_diversity_weak (Multi-hop entity-diversity counts are usually weak)
- family: Paths of three or more hops from a wechat_pay_account through identity and selected transaction relations, ending in a count or existence measure of distinct entity types.
- distribution: Similar entries in the frozen source sample are predominantly weak; type diversity alone has limited discrimination.
- bias: Evaluators may overestimate these features when equating deeper or more heterogeneous neighborhoods with stronger risk.
- basis: Counting entity existence or type diversity does not aggregate fund behavior or confirmed risk markers, so its connection to fraud is indirect.
- action: Use low-IV estimates and moderate-to-low confidence unless the plan includes stronger behavioral evidence.
- avoid: Do not apply this rule when the feature aggregates amounts, fund counterparties, or confirmed risk markers.
- boundary: Paths of at most two hops and direct single-hop amount aggregations belong to other families.

### pure_identity_chain_depth_weak (Pure identity-chain depth is usually weak)
- family: Acyclic paths composed only of wechat_pay_account, individual_customer, and natural_person identity bindings, returning depth or existence rather than cluster behavior.
- distribution: Comparable three- and four-hop pure identity paths in the frozen sample are generally weak.
- bias: Nested identity presentation can make these features appear more concealed and predictive than their measured discrimination supports.
- basis: A pure identity path establishes association but does not measure funds, coordinated account behavior, or confirmed risk.
- action: Keep IV low and reduce confidence when the only evidence is additional identity depth.
- avoid: Mixed paths containing transfer or withdrawal edges require a different calibration family.
- boundary: Same-identity cluster size and activity counts are distinct from pure path depth.

### identity_fund_mixed_chain_strong (Identity-fund mixed chains are often strong)
- family: Paths combining wechat_pay_account-to-individual_customer/natural_person identity edges with transfer or withdrawal edges, returning mixed-chain depth or remote account counts.
- distribution: Several comparable mixed paths in the frozen sample have strong measured discrimination.
- bias: Evaluators may underestimate short three- or four-hop paths by overlooking the importance of crossing from identity into fund behavior.
- basis: Remote fund counterparties and mixed-path depth measure coordinated account usage and layered fund movement more directly than identity alone.
- action: Consider higher IV and confidence when the executed plan genuinely covers both identity and fund segments.
- avoid: Pure identity paths do not qualify.
- boundary: A single fund edge without meaningful identity continuation should be assessed separately.

### same_realname_fund_aggregation_strong (Same-identity fund aggregation is often strong)
- family: Locate the target's natural_person or individual_customer, reconnect the same-identity account cluster, and aggregate transfer or withdrawal amounts or ratios across that cluster.
- distribution: Comparable same-identity fund aggregations in the frozen sample are predominantly strong.
- bias: Single-account analogies can cause underestimation because they miss group-level financial scale.
- basis: Cluster-level fund distribution reflects coordinated behavior that is invisible when each account is evaluated independently.
- action: Increase IV and confidence when cluster reconstruction and fund aggregation are explicit and executable.
- avoid: Counting cluster accounts without fund behavior belongs to a weaker family.
- boundary: Bill-deletion event proportions within the same cluster are related but use a different measured object.

### same_identity_cluster_count_mid (Same-identity cluster counts are often medium)
- family: Count merchant_pay_account, individual_customer, natural_person, bank_card, phone_number, or wechat_pay_account nodes connected through a shared identity cluster without fund aggregation or behavioral-event evidence.
- distribution: Similar frozen-source entries are commonly medium rather than strong.
- bias: Large cluster size can induce overestimation even when no behavioral or risk-marker evidence is present.
- basis: Identity aggregation scale is informative but less specific than coordinated funds or confirmed markers.
- action: Anchor estimates in the medium range and reduce high predictions unless stronger evidence is present.
- avoid: Cluster-level fund ratios or bill-deletion proportions require separate calibration.
- boundary: Singletons and trivial two-node clusters should not be treated as established coordinated groups.

### intermediary_reuse_weak_mid (Identity-intermediary reuse counts range from weak to medium)
- family: Count shared individual_customer or natural_person intermediaries reached through identity relations from multiple wechat_pay_account nodes.
- distribution: Frozen-source examples span weak and medium tiers, with weak outcomes more common.
- bias: Shared-mediator counts can be overestimated as direct evidence of coordinated fraud.
- basis: Without fund activity, intermediary reuse has limited and context-dependent discrimination.
- action: Start from a weak-to-medium estimate and avoid high confidence without corroborating behavior.
- avoid: Paths extending to fund activity require another family.
- boundary: Reuse means at least two accounts share the intermediary; do not infer risk from one connection.

### fund_transfer_topology_strong (Multi-layer fund topology is often strong)
- family: Multi-hop transfer structures measuring upstream convergence, downstream dispersion, remote-layer size, or aggregated amounts.
- distribution: Comparable multi-layer fund topologies in the frozen sample are often strong.
- bias: Single-layer analogies can underestimate the importance of continued fund propagation.
- basis: Many-to-one convergence and one-to-many dispersion across several hops directly measure layered money movement.
- action: Consider higher IV when the plan reaches and aggregates the declared remote layers.
- avoid: Simple one-hop send/receive totals without multi-hop continuation require separate calibration.
- boundary: Structures confined to two hops and lacking remote propagation do not satisfy this family.
