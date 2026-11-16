### shared_identity_account_cluster (Shared-identity account cluster)
- description: The target wechat_pay_account reaches a natural_person or individual_customer through an identity edge and reconnects to multiple wechat_pay_account nodes belonging to that subject. This is a distinct structure because coordinated multi-account control is uncommon for independent users.
- structure: wechat_pay_account -> natural_person or individual_customer through cre_id or find_sys_no, then back to other wechat_pay_account nodes through the reverse identity relation.
- substructures: An outgoing identity edge from the target; at least two distinct accounts connected to the same subject; expansion from the subject to its accounts.
- clues: natural_person or individual_customer is shared by multiple wechat_pay_account nodes, with identity relations cre_id or find_sys_no and no fund, operation-behavior, or object edge in the defining path.
- risk: Multiple accounts under one identity indicate shared control or identity reuse. Ordinary users generally control few accounts, while coordinated groups can form a structurally larger identity cluster.
- boundary: Paths containing fund edges belong to a hybrid paradigm. Undisclosed entity types are excluded. Merchant super-administrator structures are excluded.
- match_clues: cre_id, find_sys_no, natural_person, individual_customer, wechat_pay_account, terminal=wechat_pay_account, first_hop=identity, has_identity=true, has_fund=false, has_operation=false, has_object=false

### fund_pass_through_chain (Fund pass-through chain)
- description: The target participates in transfer or withdrawal relations that continue through multiple accounts or cards, representing layered money movement rather than an isolated payment.
- structure: Start at wechat_pay_account, follow personal_wechat_pay_transfer_payer_side or personal_wechat_pay_transfer_payee_side, or wechat_pay_withdrawal_to_bank_card, to direct counterparties, then continue through two or three fund hops to accounts or bank_card nodes.
- substructures: A first-hop fund counterparty; continued fund edges from that counterparty; remote fund nodes within the hop budget.
- clues: Multiple transfer counterparties, transfer or withdrawal relations, and possible convergence followed by dispersion. Do not assume an amount threshold.
- risk: Multi-hop fund chains can implement many-to-one collection, one-to-many distribution, or pass-through layering. Ordinary accounts typically have shorter and narrower fund paths.
- boundary: Identity edges make the path hybrid. Pure identity paths are excluded.
- match_clues: personal_wechat_pay_transfer_payer_side, personal_wechat_pay_transfer_payee_side, wechat_pay_withdrawal_to_bank_card, wechat_pay_account, bank_card, payer, payee, withdraw, terminal=wechat_pay_account, first_hop=fund, has_fund=true, has_identity=false, has_operation=false, has_object=false

### identity_to_fund_hybrid (Identity-to-fund hybrid)
- description: The path begins with identity sharing and then reaches fund activity from accounts in the same identity cluster, connecting coordinated control with money movement.
- structure: wechat_pay_account -> natural_person or individual_customer -> related wechat_pay_account -> fund counterparty or bank_card through a public payer_side or payee_side relation.
- substructures: Identity anchor; reverse reconnection to related accounts; outgoing or incoming fund edges from those accounts.
- clues: first_hop=identity with a later fund edge, multiple accounts on one subject, and transfer or withdrawal activity in the cluster.
- risk: A shared-identity cluster with external fund flow can conceal coordinated collection or distribution. Cluster-level activity differs structurally from an isolated account.
- boundary: A fund edge is mandatory and the first hop must be identity. Bill-deletion operations belong to the public behavior-event paradigm.
- match_clues: cre_id, find_sys_no, personal_wechat_pay_transfer_payer_side, personal_wechat_pay_transfer_payee_side, natural_person, wechat_pay_account, payer, payee, terminal=wechat_pay_account, first_hop=identity, has_identity=true, has_fund=true, has_operation=false, has_object=false

### fund_to_identity_hybrid (Fund-to-identity hybrid)
- description: The path begins with a fund counterparty and then reaches that counterparty's identity subject, exposing identity reuse behind apparently independent transactions.
- structure: wechat_pay_account -> counterparty through a public payer_side or payee_side relation, then counterparty -> individual_customer or natural_person through find_sys_no or cre_id.
- substructures: First-hop fund relation; identity edge from the counterparty; shared subject between multiple parties.
- clues: first_hop=fund with a later identity edge, especially when several payers converge through one payee and share an identity subject.
- risk: Funds moving among accounts controlled by one identity can represent internal circulation disguised as independent trade.
- boundary: An identity edge is mandatory and the first hop must be fund. Operation-behavior edges are excluded.
- match_clues: personal_wechat_pay_transfer_payer_side, personal_wechat_pay_transfer_payee_side, find_sys_no, cre_id, individual_customer, wechat_pay_account, payer, payee, terminal=individual_customer, first_hop=fund, has_identity=true, has_fund=true, has_operation=false, has_object=false

### identity_with_bill_deletion (Identity cluster with bill-deletion event)
- description: An identity-linked account reaches a bill-deletion event. The behavior may change interpretation of the controlled cluster, but it is not itself proof of fraud.
- structure: wechat_pay_account -> identity subject -> related account -> wechat_pay_account -> wechat_pay_bill_deletion_event through account_uin.
- substructures: Identity anchor; account cluster; a bill-deletion event attached to a member.
- clues: first_hop=identity plus wechat_pay_bill_deletion on the identity network.
- risk: Repeated bill deletion among accounts sharing an identity may warrant review only when supported by other observed behavior.
- boundary: Fund edges are excluded for mutual exclusivity. Direct target-to-event paths without an identity intermediary are excluded.
- match_clues: cre_id, find_sys_no, wechat_pay_bill_deletion_event, wechat_pay_bill_deletion, natural_person, individual_customer, terminal=wechat_pay_bill_deletion_event, first_hop=identity, has_identity=true, has_fund=false, has_operation=true, has_object=false
