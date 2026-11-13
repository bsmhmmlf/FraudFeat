### shared_identity_transfer_amount (Shared-identity transfer amount)
- description: For a cluster reconnected through cre_id, aggregate outgoing transfer amounts from all related accounts to measure cluster-level distribution strength.
- paradigm: Shared-identity account cluster (shared_identity_account_cluster)
- structure: Target wechat_pay_account -> natural_person -> related wechat_pay_account nodes -> outgoing personal_wechat_pay_transfer_payer_side edges.
- feature_form: amount; aggregate funds rather than merely counting accounts to capture coordinated financial scale.
- discrimination: An ordinary identity usually controls few accounts with limited aggregate outflow; coordinated multi-account clusters can produce much larger group-level outflow.
- example: Several accounts under one identity collectively pay many counterparties, while an ordinary user operates one account with sparse payments.
- risk: Multiple accounts under one controller can divide fund distribution across accounts, making cluster-level aggregation more informative than isolated balances.
- boundary: Requires an identity edge and at least one additional account; a singleton cluster has no aggregation value.

### shared_identity_collection_ratio (Shared-identity collection ratio)
- description: Within a shared-identity cluster, calculate the fraction of accounts that exhibit the defined large-incoming-transfer behavior to measure collection penetration.
- paradigm: Shared-identity account cluster (shared_identity_account_cluster)
- structure: Related accounts receive funds through personal_wechat_pay_transfer_payee_side; divide the qualifying subset by the full cluster.
- feature_form: ratio; the proportion reveals whether collection is distributed across the identity network.
- discrimination: A coordinated cluster can contain several collection accounts, whereas ordinary small clusters do not show systematic role separation.
- example: Two of three accounts under one identity repeatedly receive upstream transfers.
- risk: Multiple collection endpoints under one identity can implement distributed fund aggregation.
- boundary: Requires at least two cluster accounts and an explicitly observed threshold when large-transfer filtering is used.

### fund_chain_remote_count (Remote fund-chain account count)
- description: Expand transfer or withdrawal relations through three hops and count remote wechat_pay_account nodes at the deepest level.
- paradigm: Fund pass-through chain (fund_pass_through_chain)
- structure: Target -> first-hop counterparty -> second-hop intermediary -> third-hop remote wechat_pay_account.
- feature_form: count; remote breadth captures the reach and complexity of layered fund movement.
- discrimination: Ordinary payment paths usually terminate quickly, while pass-through networks can continue to many remote accounts.
- example: Three transfer hops reveal a broad set of previously unseen accounts rather than a small set of known contacts.
- risk: Multi-layer transfer and withdrawal paths can implement layered laundering and pass-through circulation.
- boundary: Requires fund paths whose deepest observed endpoint is an account; immediate card termination is a different structure.

### fund_pass_through_ratio (Pass-through inflow/outflow ratio)
- description: When the target both receives and sends transfers, compute a clearly defined ratio between outgoing and incoming totals to characterize fund retention.
- paradigm: Fund pass-through chain (fund_pass_through_chain)
- structure: Aggregate incoming and outgoing transfer amounts in separate anchored steps and reduce them with final.ratio.
- feature_form: ratio; balance between flows can reveal pass-through behavior that absolute totals alone cannot.
- discrimination: Pass-through accounts may show closely matched inflow and outflow at substantial scale, while ordinary accounts retain meaningful differences.
- example: An account receives and forwards nearly equal totals within the same observed behavior window.
- risk: Collection followed by immediate distribution is consistent with a pass-through channel.
- boundary: Requires both directions and a nonzero, explicitly defined denominator; do not infer temporal causality without timestamps.

### identity_fund_hybrid_depth (Identity-to-fund hybrid depth)
- description: Measure the longest simple path from the target through an identity subject and related account to a fund counterparty.
- paradigm: Identity-to-fund hybrid (identity_to_fund_hybrid)
- structure: wechat_pay_account -> natural_person through cre_id -> related wechat_pay_account -> transfer counterparty.
- feature_form: count of hops; depth captures how far identity-linked fund paths extend.
- discrimination: Ordinary identity chains often stop before fund distribution, while coordinated clusters can continue into multiple fund counterparties.
- example: One related account under the same identity continues to an external transfer, producing a three-hop mixed path.
- risk: Identity reuse combined with onward fund movement connects shared control to distribution behavior.
- boundary: Requires both identity and fund edges; pure identity clusters are excluded.

### fund_identity_reuse_count (Fund-counterparty identity reuse count)
- description: From a shared payee, reconnect other payers and count distinct individual_customer or natural_person subjects linked to those payers to expose identity reuse around fund convergence.
- paradigm: Fund-to-identity hybrid (fund_to_identity_hybrid)
- structure: Target -> payee -> other payers -> their individual_customer or natural_person subjects through find_sys_no or cre_id.
- feature_form: count; distinct identity subjects measure the breadth of the convergence network.
- discrimination: Independent counterparties usually have dispersed identities; coordinated groups can show unusual sharing or concentration around one payee.
- example: Many payers use one payee and also share an identity subject, revealing coordinated control.
- risk: A shared payee intermediary can combine fund aggregation with identity reuse. Use only observed graph identifiers in the executable logic.
- boundary: Requires a fund first hop plus reverse fund and identity relations; a single transfer is insufficient.

### identity_cluster_bill_deletion_ratio (Identity-cluster bill-deletion ratio)
- description: Within an identity-linked account cluster, calculate the fraction of accounts connected to a wechat_pay_bill_deletion_event through account_uin.
- paradigm: Identity cluster with bill-deletion event (identity_with_bill_deletion)
- structure: Shared-identity accounts -> wechat_pay_account -> wechat_pay_bill_deletion_event relations; divide accounts with the event by cluster size.
- feature_form: ratio; concentration of bill-deletion behavior across a shared-identity cluster.
- discrimination: Concentrated bill deletion may distinguish a coordinated cluster from incidental credential sharing; the event is not a fraud label.
- example: Two of five accounts under one identity have bill-deletion events.
- risk: Shared control makes repeated bill deletion relevant to review of the whole cluster, subject to supporting evidence.
- boundary: Requires observed wechat_pay_bill_deletion_event events and a nonzero denominator; absence yields no such behavioral evidence.

### bound_customer_natural_person_flag (Bound-customer natural-person flag)
- description: Check whether the individual_customer directly bound through find_sys_no has a cre_id binding to a natural_person.
- paradigm: Shared-identity account cluster (shared_identity_account_cluster)
- structure: Target wechat_pay_account -> individual_customer -> natural_person through the two public identity bindings.
- feature_form: flag; existence of the public identity relation is the measured fact.
- discrimination: A complete public identity chain differs structurally from a customer binding without a linked natural_person.
- example: The target's bound customer connects to a natural_person with the same cre_id.
- risk: The feature captures public identity continuity, not an undisclosed entity or label.
- boundary: Requires find_sys_no and cre_id relations; absence of the relation is not evidence of fraud.
