### expand_strong_identity_media_first (Prioritize customer and natural-person anchors)
- paradigms: pure_identity_multi_hop_wechat_derivation,pure_identity_multi_hop_natural_person_derivation,dual_medium_alt_wechat_derivation
- signal: The observed center or a reachable account has outgoing wechat_pay_account_bind_individual_customer or wechat_pay_account_bind_natural_person edges.
- why: The source trajectory expand_strong_identity_media_first repeatedly found that identity records expose account associations that a transfer-only view misses. The identity edge is a bridge, not a fraud verdict.
- do: Follow the binding out to individual_customer or natural_person, then follow the same relation in to associated accounts. Record the endpoint types and directions before extending another identity medium.
- avoid: Do not stop after counting direct identity records, and do not assume every account returned by the reverse binding is controlled by the same person.
- boundary: Skip identity re-expansion when it produces only the already-visited seed and no new account or second identity medium. Legitimate shared records require downstream behavioral corroboration.

### trace_transfer_to_find_relay (Complete a transfer relay before naming a pattern)
- paradigms: pure_fund_chain_layered_transfer,identity_anchored_transfer_relay_reserve_wechat
- signal: A reachable account has payer-side recipients whose outgoing personal_wechat_pay_transfer_payee_side edges reveal their own payers.
- why: trace_transfer_to_find_relay showed that stopping at the first recipient confuses a direct payment with a shared-recipient relay. The second relation reveals whether multiple payers converge on the same recipients.
- do: Traverse payer_side out to recipients, payee_side in to co-payers, and then payee_side out from those co-payers when receipts are measured. Keep each step's output and economic role separate.
- avoid: Do not describe two stored reversals of one relation as two consecutive downstream transfers, and do not sum an intermediate edge when the terminal receipt amount is intended.
- boundary: If the second hop returns only the previous account set or no new edge, record a direct-payment neighborhood rather than a relay.

### expand_deep_dual_bound_reserve (Require both identity bindings only after locating the deeper cluster)
- paradigms: identity_anchored_transfer_relay_reserve_wechat,dual_medium_alt_wechat_derivation
- signal: A natural-person/account expansion and a transfer bridge lead to another natural-person record whose associated accounts also have customer bindings.
- why: expand_deep_dual_bound_reserve distinguishes a dual-qualified terminal account set from the much broader unqualified reserve-account count.
- do: Follow wechat_pay_account_bind_natural_person out/in, complete the observed transfer hop, reach the second natural_person, reverse its binding to accounts, and test outgoing wechat_pay_account_bind_individual_customer on those terminal accounts.
- avoid: Do not apply the customer-binding requirement to the first account set or replace a dual-binding ratio with an unqualified account count.
- boundary: The structure is unavailable when either identity relation or the transfer bridge is absent. A filter changes the hypothesis and its calibration family.

### discard_isolated_leaf_noise (Discard only non-bridging peripheral expansions)
- paradigms: pure_fund_chain_layered_transfer,payee_view_identity_bridge_wechat_reach
- signal: An expanded leaf has no unvisited outgoing identity, transfer, merchant, or instrument edge and cannot contribute to the current hypothesis chain.
- why: discard_isolated_leaf_noise preserved traversal budget by removing leaves after testing their expansion value instead of treating every newly visited vertex as equally useful.
- do: Check both stored directions, available relation types, and whether the vertex is required by an existing evidence chain before discarding its pending expansion group.
- avoid: Never discard a customer, natural-person, merchant, or bank-card endpoint that supplies the terminal entity or a required filter merely because its own fan-out is small.
- boundary: Keep a leaf when the proposed statistic is its existence, degree, binding, or amount. Zero expandable neighbors does not mean zero feature value.

### avoid_assume_normal_zero_on_identity_cluster (Treat broad identity clusters as calibratable)
- paradigms: pure_identity_multi_hop_wechat_derivation,dual_medium_alt_wechat_derivation
- signal: A shared customer or natural-person record reaches several ordinary-looking accounts with normal authentication timestamps and no observed transfer anomaly.
- why: avoid_assume_normal_zero_on_identity_cluster corrected the over-general assumption that every large identity cluster is almost absent among normal accounts. Reachability is not measured discrimination.
- do: Preserve the candidate but propose a specific count, ratio, or flag; ask the evaluator to compare its confidence and IV with the exact endpoint, depth, and filter.
- avoid: Do not discard the structure solely because normal hits may exist, and do not inflate IV solely because the cluster is large or identity-mediated.
- boundary: Shared family, employer, or business records can create legitimate clusters. Status and authentication properties must be observed before being used as filters.

### reverse_anchor_merchant_upstream_control (Inspect payers behind a reached merchant)
- paradigms: identity_anchored_controlled_wechat_admin_merchant_receive,pure_fund_chain_layered_transfer
- signal: Navigation reaches a merchant_pay_account with outgoing merchant_payment_receipt_payee_side edges, and those payers expose additional identity or transfer neighborhoods.
- why: reverse_anchor_merchant_upstream_control contributed the reverse-anchor idea: start from a reached business endpoint and inspect its receipt payers instead of treating the merchant as a terminal label.
- do: Traverse merchant_payment_receipt_payee_side out from merchant to payer accounts, then inspect those payers' customer bindings and payee-side personal transfers using their stored directions.
- avoid: Do not use a prohibited risk label to reach the merchant, and do not assume that every payer belongs to the seed's identity-associated account set.
- boundary: Ordinary merchants have many legitimate payers. Keep payer diversity, receipt amount, and subsequent identity association as separate candidate statistics.

### expand_controlled_wechat_admin_merchant (Complete administration before aggregation)
- paradigms: identity_anchored_controlled_wechat_admin_merchant_receive
- signal: An identity-associated account has outgoing personal_wechat_pay_account_is_merchant_super_administrator and the reached merchant has receipt activity.
- why: expand_controlled_wechat_admin_merchant separated administration-only existence from administered-merchant receipt amount, two features that are often collapsed into a vague merchant-association claim.
- do: Complete the customer/account bridge, follow the administrator relation out, verify merchant_pay_account, then choose either exists on merchants or sum merchant_payment_receipt_payee_side total_amount.
- avoid: Do not replace the administrator relation with direct merchant payment, and do not infer fraudulent merchant status from the management role.
- boundary: If no merchant is reached, return the appropriate empty-set result and reconsider the chain. If no receipt edge exists, administration may still support a flag but not an amount.
