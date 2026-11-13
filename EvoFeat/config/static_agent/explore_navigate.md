### shared_identity_reconnect (Reconnect accounts through shared identity)
- description: When the target has an outgoing identity edge to a natural_person or individual_customer, reverse the relation from that subject to find other wechat_pay_account nodes and complete the shared-identity cluster.
- signal: A first-hop cre_id or find_sys_no identity edge reaches a subject connected to at least two distinct wechat_pay_account nodes.
- action: Expand all relevant accounts from the subject through reverse identity edges, then inspect their fund and operation relations.
- paradigms: Shared-identity account cluster (shared_identity_account_cluster); Identity cluster with bill-deletion event (identity_with_bill_deletion); Identity-to-fund hybrid (identity_to_fund_hybrid)
- example: Account A reaches a natural_person through cre_id. Reverse expansion finds accounts B and C under the same person, so B and C join the investigation.
- risk: Several accounts controlled through one credential indicate shared control and should be interpreted as one coordinated set rather than independent users.
- boundary: If the subject connects only to the target, no account cluster has been established.

### fund_pass_through_expand (Expand a fund pass-through chain)
- description: When transfer or withdrawal edges reach a counterparty that has further fund relations, continue through the available hops to complete the layered chain.
- signal: The target has personal_wechat_pay_transfer_payer_side/payee_side or wechat_pay_withdrawal_to_bank_card edges, and a counterparty can reveal additional fund neighbors.
- action: Expand second- and third-hop fund nodes in justified directions, then inspect identity bindings to establish ownership.
- paradigms: Fund-to-identity hybrid (fund_to_identity_hybrid); Identity-to-fund hybrid (identity_to_fund_hybrid); Fund pass-through chain (fund_pass_through_chain)
- example: Account X pays Y, and Y subsequently withdraws or transfers to Z. Following that continuation completes a multi-hop pass-through path.
- risk: Layered movement can obscure the original controller through collection, dispersion, or pass-through behavior.
- boundary: A counterparty with no further fund edge represents a single transaction rather than a multi-hop chain.

### identity_to_fund_extend (Extend an identity cluster into fund activity)
- description: After reconnecting accounts through a shared identity, inspect whether those accounts have fund edges to complete an identity-to-fund hybrid.
- signal: A first-hop identity subject connects multiple wechat_pay_account nodes, at least one of which has transfer or withdrawal relations.
- action: Expand fund counterparties from the related accounts and inspect direction, amounts, and counterparties without inventing thresholds.
- paradigms: Shared-identity account cluster (shared_identity_account_cluster); Identity-to-fund hybrid (identity_to_fund_hybrid); Fund pass-through chain (fund_pass_through_chain)
- example: Accounts A and B share an identity subject, and B pays C. Expanding B's fund edge connects the identity cluster to C.
- risk: Coordinated accounts under one controller can distribute funds across paths that appear unrelated at the single-account level.
- boundary: If no related account has a fund edge, keep the structure as a pure identity cluster.

### fund_to_identity_backtrack (Backtrack from a fund counterparty to identity)
- description: From a transfer counterparty, inspect individual_customer or natural_person bindings to determine whether the parties share an identity subject.
- signal: A first-hop fund counterparty also has find_sys_no or cre_id identity edges.
- action: Expand the counterparty's identity subject and compare it with the target's subject; record independent identities to avoid false connections.
- paradigms: Shared-identity account cluster (shared_identity_account_cluster); Identity cluster with bill-deletion event (identity_with_bill_deletion); Fund-to-identity hybrid (fund_to_identity_hybrid)
- example: X receives funds from Y. Both X and Y bind the same individual_customer, revealing internal circulation within one controlled identity.
- risk: Apparently independent counterparties belonging to one controller can disguise self-circulation as external trade.
- boundary: If no identity edge is available, defer the identity conclusion.

### identity_cluster_bill_deletion (Inspect bill-deletion behavior in an identity cluster)
- description: After establishing an identity cluster, inspect every member for the public wechat_pay_account -> wechat_pay_bill_deletion_event relation.
- signal: A related account has a wechat_pay_bill_deletion edge.
- action: Compare cluster members for this observed behavior without interpreting it as a risk label.
- paradigms: Shared-identity account cluster (shared_identity_account_cluster); Identity cluster with bill-deletion event (identity_with_bill_deletion); Identity-to-fund hybrid (identity_to_fund_hybrid)
- example: Accounts A and B share an identity. B has a bill-deletion event, which may be evaluated alongside other behavior.
- risk: Repeated bill deletion across a shared-identity network can justify further review only with supporting evidence.
- boundary: Without a bill-deletion edge, keep the structure as an identity cluster without that behavior.

### same_cluster_media_reuse (Inspect card and phone reuse within an identity cluster)
- description: When multiple accounts share an identity, inspect whether they also share bank_card nodes or phone_number nodes.
- signal: Related accounts connect to the same bank_card or phone_number through observed public relations.
- action: Deduplicate shared objects, expand their other connected accounts, and complete the reuse network.
- paradigms: Shared-identity account cluster (shared_identity_account_cluster); Identity cluster with bill-deletion event (identity_with_bill_deletion); Identity-to-fund hybrid (identity_to_fund_hybrid)
- example: Two accounts under one individual_customer bind the same bank_card, providing independent evidence of coordinated control.
- risk: Independent users usually have separate bank_card and phone_number nodes; shared objects can strengthen the shared-control interpretation.
- boundary: Disjoint object sets indicate identity sharing without card or phone reuse.
