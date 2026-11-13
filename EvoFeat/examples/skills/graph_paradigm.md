### pure_identity_multi_hop_wechat_derivation (Repeated customer-mediated account expansion)
- description: Rewritten from the learned pure_identity_multi_hop_wechat_derivation trajectory. Repeated customer/account bridges expose an account set beyond the seed's directly associated accounts without relying on a transfer edge.
- structure: Follow wechat_pay_account_bind_individual_customer out to a customer, in to associated accounts, and repeat that out/in bridge at least once. The terminal entity is wechat_pay_account.
- substructures: Initial customer record; first associated-account set; a second customer bridge; terminal account set. The fourth-hop count differs semantically from a one-bridge count even when both use the same relation.
- clues: At least two customer/account re-expansions, multiple accounts attached to a customer record, and no transfer, withdrawal, recharge, or merchant-payment edge.
- risk: Repeated identity-mediated expansion can reveal a broad connected account cluster. Ordinary associations may also produce nonzero counts, so depth and terminal identity must remain explicit.
- boundary: Do not include natural-person alternation, a single direct binding, or any path containing a fund edge. Counts and ratios may share the structure but remain different statistics.
- match_clues: wechat_pay_account_bind_individual_customer, repeated customer bridge, terminal=wechat_pay_account, first_hop=identity, has_identity=yes, has_fund=no, value=count ratio

### pure_identity_multi_hop_natural_person_derivation (Identity expansion ending at natural persons)
- description: Adapted from pure_identity_multi_hop_natural_person_derivation. An identity/account walk ends at natural-person records rather than payment accounts, changing both the measured population and the false-positive boundary.
- structure: Traverse customer or natural-person bindings between accounts and identity records for three or more hops, then terminate with outgoing wechat_pay_account_bind_natural_person.
- substructures: Initial identity anchor; intermediate associated accounts; a second identity bridge; terminal natural_person records.
- clues: Both account and subject entities appear, no monetary edge is present, and the terminal entity is natural_person rather than another account.
- risk: The terminal set describes the breadth of natural-person records connected through the observed identity chain. It must not be interpreted as proof of distinct real-world controllers.
- boundary: A terminal account belongs to account derivation; any transfer edge belongs to a mixed identity-transfer paradigm. Family or household identity sharing can create legitimate matches.
- match_clues: wechat_pay_account_bind_natural_person, wechat_pay_account_bind_individual_customer, terminal=natural_person, first_hop=identity, has_identity=yes, has_fund=no, value=count ratio

### dual_medium_alt_wechat_derivation (Alternating customer and natural-person expansion)
- description: Rewritten from dual_medium_alt_wechat_derivation. Both identity record types occur in the same account-expansion path, making the topology distinct from repeated use of either medium alone.
- structure: Start with either customer or natural-person binding, reverse the same relation to accounts, switch to the other identity relation, and reverse it to a terminal account set. A four-hop natural-person/customer alternation is the canonical example.
- substructures: First identity denominator; first associated accounts; second identity medium; terminal account numerator or count.
- clues: Both wechat_pay_account_bind_individual_customer and wechat_pay_account_bind_natural_person occur, with account vertices between them and no fund edge.
- risk: Alternating media measures cross-record expansion rather than the size of one identity cluster. Ratios should state whether the denominator is initial customers or initial natural persons.
- boundary: One-medium repeated bridges, terminal identity records, and paths containing transfers are separate paradigms. Legitimate shared identity records may still create expansion.
- match_clues: dual identity medium, wechat_pay_account_bind_individual_customer, wechat_pay_account_bind_natural_person, terminal=wechat_pay_account, first_hop=identity, has_identity=yes, has_fund=no, value=count ratio

### pure_identity_multi_hop_bankcard_flag (Deep identity walk with bank-card endpoint)
- description: Adapted from pure_identity_multi_hop_bankcard_flag. A repeated identity expansion ends at a bound bank card and is reduced to an existence flag instead of an account count.
- structure: Complete two or more customer/account bridges, then follow outgoing wechat_pay_account_bind_bank_card from terminal accounts to bank_card endpoints and apply exists.
- substructures: Repeated customer/account expansion; final account set; terminal instrument-binding edge; binary endpoint-presence reduction.
- clues: The path contains identity relations and exactly one terminal binding relation, no transfer or withdrawal edge, and terminal=bank_card.
- risk: The feature records whether an instrument association exists after a deep identity walk. It says nothing about transaction amount, cash-out, or who controls the card.
- boundary: Do not substitute wechat_pay_withdrawal_to_bank_card for a binding relation. A count of cards or a shorter binding path is a different feature.
- match_clues: bank_card, wechat_pay_account_bind_bank_card, terminal=bank_card, first_hop=identity, has_identity=yes, has_fund=no, value=flag

### pure_fund_chain_layered_transfer (Transfer-only alternating or layered relay)
- description: Rewritten from pure_fund_chain_layered_transfer. The complete path contains only personal transfer relations; direction, hop count, and final reduction determine the feature.
- structure: Chain personal_wechat_pay_transfer_payer_side and personal_wechat_pay_transfer_payee_side with explicit out/in directions. A payee out/payer out/payer out amount path and a payee out/in/out/in count path are not interchangeable.
- substructures: Initial payer or recipient set; one or more transfer relays; terminal account count, amount sum, or layer-to-layer ratio.
- clues: Only wechat_pay_account vertices and personal transfer relations occur; no customer, natural-person, bank-card, or merchant relation mediates the walk.
- risk: Transfer layering can describe counterparty reach or dispersion, but ordinary business and household transfers also form chains. Depth alone is not evidence of fraud.
- boundary: Adding any identity edge changes the paradigm. Outgoing payee-side traversal finds payers of the current recipients; incoming payee-side traversal finds recipients of the current payers.
- match_clues: personal_wechat_pay_transfer, pure transfer relay, terminal=wechat_pay_account, first_hop=fund, has_identity=no, has_fund=yes, value=count ratio amount

### payee_view_identity_bridge_wechat_reach (Transfer-first identity bridge)
- description: Adapted from payee_view_identity_bridge_wechat_reach. A transfer edge is followed by an identity bridge, so the first-hop role is different from identity-first relay features.
- structure: Follow outgoing personal_wechat_pay_transfer_payee_side from the seed recipient to a payer, traverse that payer's customer or natural-person binding, and reverse it to associated accounts. Optional large-transfer and self-exclusion filters belong in logic and the plan.
- substructures: Seed receipt; upstream payer; payer identity record; accounts associated with that identity; optional amount threshold and terminal qualification.
- clues: The first relation is payee-side transfer, an identity relation appears only after it, and the terminal objects are accounts or identity records.
- risk: The pattern combines a payment relationship with a wider identity-associated account set. Small ordinary transfers and shared family accounts can produce false positives unless filters are observed and justified.
- boundary: Do not relabel an identity-first path as this paradigm. A second transfer relay before the identity bridge changes hop count and calibration family.
- match_clues: payee identity bridge, personal_wechat_pay_transfer_payee_side, wechat_pay_account_bind_individual_customer, terminal=wechat_pay_account, first_hop=fund, has_identity=yes, has_fund=yes, value=count

### identity_anchored_transfer_relay_reserve_wechat (Identity-first relay to associated accounts)
- description: Rewritten from identity_anchored_transfer_relay_reserve_wechat. An initial identity bridge reaches accounts, a transfer relation reaches another account neighborhood, and a later identity bridge exposes a deeper account set.
- structure: Use customer/account expansion, at least one explicitly directed payer-side or payee-side transfer step, and a subsequent identity/account expansion ending at wechat_pay_account.
- substructures: Initial identity-associated accounts; transfer bridge; second identity cluster; optional requirement that terminal accounts have both customer and natural-person bindings.
- clues: Identity edges occur before and after a transfer edge; terminal accounts, not raw transfer counterparties, are counted or normalized.
- risk: This topology separates a payment bridge from the deeper account cluster it reaches. A dual-binding qualification is stricter than an unqualified account count and must not silently replace it.
- boundary: If the final hop is an amount edge, use the corresponding amount paradigm. If no second identity expansion exists, this is only identity-first payment fan-out.
- match_clues: identity transfer reserve, personal_wechat_pay_transfer, repeated identity bridge, terminal=wechat_pay_account, first_hop=identity, has_identity=yes, has_fund=yes, value=count ratio

### identity_fund_relay_derived_natural_person (Identity-transfer walk ending at natural persons)
- description: Adapted from identity_fund_relay_derived_natural_person. Identity expansion and a mixed-direction transfer relay terminate at natural-person records, providing a different endpoint from account fan-out.
- structure: Expand through an initial natural-person or customer relation, follow payer-side out and payee-side in transfer steps, then finish with outgoing wechat_pay_account_bind_natural_person.
- substructures: Initial identity cluster; first payment recipients; recipients of those accounts via incoming payee-side edges; terminal natural-person records.
- clues: At least one identity and one transfer relation occur, the terminal entity is natural_person, and the last hop is identity rather than transfer.
- risk: The feature measures identity reach after a payment relay. It does not establish a fund-return loop or common control of every terminal person record.
- boundary: Terminal accounts belong to reserve-account expansion; terminal transfer amounts belong to monetary aggregation. Missing either transfer direction creates a different topology.
- match_clues: identity fund relay, natural person endpoint, personal_wechat_pay_transfer, wechat_pay_account_bind_natural_person, terminal=natural_person, first_hop=identity, has_identity=yes, has_fund=yes, value=count ratio

### identity_anchored_controlled_wechat_admin_merchant_receive (Identity-associated merchant administration and receipts)
- description: Rewritten from identity_anchored_controlled_wechat_admin_merchant_receive without using the source system's risk labels. Identity-associated accounts administer merchants, and merchant receipt edges are then counted or summed.
- structure: Expand customer-associated accounts, follow outgoing personal_wechat_pay_account_is_merchant_super_administrator to merchant_pay_account, then outgoing merchant_payment_receipt_payee_side to receipt payers or directly to total_amount.
- substructures: Customer/account bridge; merchant-administration edge; merchant set; payer-account set or merchant-receipt amount. Administration without receipts is an existence variant.
- clues: A merchant administrator relation precedes a merchant payee-side receipt relation; the terminal value may be flag, payer count, or RMB-fen amount.
- risk: The path links account associations with administered business activity. Administration alone does not establish merchant fraud, and receipt payers need not belong to the initial account cluster.
- boundary: Direct payer-side merchant payment is not merchant administration. Keep merchant receipt amount distinct from personal transfer amount and from the administration-only flag.
- match_clues: merchant administration, personal_wechat_pay_account_is_merchant_super_administrator, merchant_payment_receipt_payee_side, terminal=wechat_pay_account, first_hop=identity, has_identity=yes, has_fund=yes, value=count amount flag

### identity_anchored_wechat_withdraw_bankcard (Identity-payment relay ending at withdrawals)
- description: Adapted from identity_anchored_wechat_withdraw_bankcard. An identity-associated account set pays recipients whose outgoing withdrawal edges are then aggregated.
- structure: Follow customer/account expansion, outgoing personal_wechat_pay_transfer_payer_side to recipients, and outgoing wechat_pay_withdrawal_to_bank_card from those recipients to bank cards while summing total_amount.
- substructures: Identity-associated payers; payment recipients; terminal withdrawal edges; RMB-fen amount sum. The measured amount belongs to the last hop, not the preceding transfer.
- clues: A customer binding and a personal transfer precede the withdrawal relation; the terminal edge carries the monetary aggregate.
- risk: The feature links a payment neighborhood to cash-out activity. Ordinary recipients may also withdraw funds, so the result should be interpreted with payer/recipient breadth and observed amounts.
- boundary: Bank-card binding is not withdrawal; merchant payment is not withdrawal. Reversing the withdrawal edge or summing the preceding transfer changes the feature.
- match_clues: withdrawal, wechat_pay_withdrawal_to_bank_card, personal_wechat_pay_transfer_payer_side, terminal=bank_card, first_hop=identity, has_identity=yes, has_fund=yes, value=amount count
