### deep_dual_bound_wechat_count (Alternating-identity terminal account count)
- description: Rewritten from deep_dual_bound_wechat_count. Alternate customer and natural-person bindings, reverse each identity record to its accounts, and count the terminal accounts rather than the intermediate identity records.
- references: dual_medium_alt_wechat_derivation; pure_identity_multi_hop_wechat_derivation
- paradigm: dual_medium_alt_wechat_derivation
- structure: Follow wechat_pay_account_bind_natural_person out to natural persons, in to accounts, wechat_pay_account_bind_individual_customer out to customers, and in to terminal accounts. A terminal customer-binding or natural-person qualification must be an explicit filter.
- feature_form: expected_value_type=count; final type=count over distinct terminal account IDs. Keep the four relation hops separate from any terminal where qualification.
- discrimination: Compare exact hop depth and terminal filters. Repeated identity reach may separate broad connected clusters, but direct customer binding and single-medium walks are different baselines.
- example: Candidate name deep_dual_bound_account_count. Logic states both media, both reverse expansions, the terminal qualification, and whether earlier accounts remain eligible for the count.
- risk: Broad identity-connected account sets can support review of coordinated account infrastructure when transaction or instrument evidence corroborates them.
- boundary: Do not invent a direct customer-to-natural-person relation. Shared household or business identities can produce legitimate expansion; a missing second medium makes this hypothesis inapplicable.

### dual_identity_deep_nat_count (Alternating-identity terminal natural-person count)
- description: Adapted from dual_identity_deep_nat_count. Alternate natural-person and customer media, then terminate at natural-person records reached from the final account set.
- references: pure_identity_multi_hop_natural_person_derivation; dual_medium_alt_wechat_derivation
- paradigm: pure_identity_multi_hop_natural_person_derivation
- structure: Natural-person binding out/in, customer binding out/in, then natural-person binding out. The five-hop terminal population is natural_person, not the fourth-hop accounts.
- feature_form: expected_value_type=count; five id_list steps and final count over distinct terminal natural_person IDs.
- discrimination: Compare against terminal-account expansion and one-medium natural-person counts. A larger account set does not imply a proportionally larger terminal identity set.
- example: Candidate name alternating_identity_terminal_natural_person_count. Logic names both intermediate media, the final account set, and the terminal natural-person binding.
- risk: The feature measures identity-record reach after alternating expansion and can reveal a broader identity neighborhood than a direct binding count.
- boundary: Do not infer distinct real-world controllers from record count alone. Missing either medium or inserting a transfer edge changes the applicable paradigm.

### cust_twice_bridge_wechat_cnt (Two customer-bridge account count)
- description: Adapted from cust_twice_bridge_wechat_cnt. Repeat the customer/account bridge twice and count fourth-hop accounts, preserving the difference from a direct or one-bridge association count.
- references: pure_identity_multi_hop_wechat_derivation
- paradigm: pure_identity_multi_hop_wechat_derivation
- structure: wechat_pay_account_bind_individual_customer out, in, out, in: account to customer, customer to accounts, account to customer, customer to terminal accounts.
- feature_form: expected_value_type=count; use four id_list steps and final count on s4. Do not subtract the seed unless logic explicitly requires exclusion.
- discrimination: The useful comparison is repeated expansion versus direct binding and one-bridge expansion. No universal count threshold is carried over from the source trajectory.
- example: Candidate name customer_double_bridge_account_count. Logic identifies the two complete bridges and the terminal distinct-account reduction.
- risk: The statistic can expose an unusually wide identity-connected account neighborhood while remaining explainable as a graph count.
- boundary: Ordinary account clusters can also be large. A ratio, terminal identity endpoint, or added transfer edge requires a different hypothesis entry.

### ratio_cre_reserve_to_cluster_wechat (Dual-qualified reserve accounts per initial cluster)
- description: Rewritten from ratio_cre_reserve_to_cluster_wechat. Use the initial natural-person-associated account set as denominator and a transfer-reached, second-identity expansion with a customer-binding qualification as numerator.
- references: identity_anchored_transfer_relay_reserve_wechat; dual_medium_alt_wechat_derivation
- paradigm: identity_anchored_transfer_relay_reserve_wechat
- structure: Natural-person binding out/in establishes the denominator. Follow an observed payer/payee transfer bridge to another account set, then natural-person binding out/in reaches the numerator; require customer binding only on terminal numerator accounts.
- feature_form: expected_value_type=ratio; numerator is distinct qualified terminal accounts, denominator is distinct initial natural-person-associated accounts, zero denominator yields 0.
- discrimination: Normalization separates relative cross-cluster expansion from raw account-set size. The dual-binding filter and transfer directions are part of the statistic and must not be dropped.
- example: Candidate name transfer_reached_dual_bound_accounts_per_initial_identity_account. Logic names both populations before specifying their ratio.
- risk: The feature can reveal a disproportionate reserve-account set connected through payment and identity bridges without relying on an absolute cluster-size cutoff.
- boundary: Family or business relationships can produce both bindings. An unqualified terminal count, a different denominator, or a missing transfer bridge is not this feature.

### sum_mule_transfer_out_amount (Identity-associated relay payment amount)
- description: Adapted from sum_mule_transfer_out_amount. Expand customer-associated accounts, reach their upstream payers with outgoing payee-side edges, then sum those payers' outgoing payer-side transfer amounts.
- references: identity_anchored_transfer_relay_reserve_wechat; pure_fund_chain_layered_transfer
- paradigm: identity_anchored_transfer_relay_reserve_wechat
- structure: Customer binding out/in; personal_wechat_pay_transfer_payee_side out from associated recipients to payers; personal_wechat_pay_transfer_payer_side out from those payers to recipients while summing terminal total_amount.
- feature_form: expected_value_type=amount; amount_unit=fen; final scalar sum over distinct terminal payer-side edges. The measured money is the last-hop outgoing payment amount.
- discrimination: Compare against the corresponding terminal count and the unanchored transfer amount. The identity prefix and change from incoming to outgoing economic direction must remain visible in logic.
- example: Candidate name customer_associated_upstream_payer_payment_amount_fen. Logic names the upstream payer set before summing its payments.
- risk: The feature measures payment dispersion around an identity-linked upstream neighborhood and can complement collection-side statistics.
- boundary: Ordinary repeat payments may have high amounts. Summing the incoming payee-side edges instead would measure receipts of the associated accounts, not the intended payer fan-out.

### payee_bridge_large_transfer_reach (Payer identity reach after a qualified receipt)
- description: Rewritten from payee_bridge_large_transfer_reach. Start from an incoming payment, inspect the payer's identity-associated accounts, and retain only filters supported by observed amounts and hypothesis logic.
- references: payee_view_identity_bridge_wechat_reach
- paradigm: payee_view_identity_bridge_wechat_reach
- structure: Outgoing personal_wechat_pay_transfer_payee_side reaches the payer of the seed recipient; outgoing customer or natural-person binding reaches the payer's identity record; incoming binding reaches associated accounts.
- feature_form: expected_value_type=count; count distinct terminal accounts after any stated large-transfer threshold and seed-exclusion rule. Thresholds are data-dependent, not universal constants.
- discrimination: The transfer-first prefix distinguishes this statistic from identity-first fan-out. Compare filtered and unfiltered variants rather than assuming every same-identity account is suspicious.
- example: Candidate name incoming_payer_identity_associated_account_count. Logic states the receipt qualification, identity medium, reverse expansion, and exclusion rule.
- risk: This pattern can show how a payment relationship connects to a wider account cluster, with filters helping separate casual small transfers from the intended neighborhood.
- boundary: Large family or business payments can still be legitimate. If the amount filter was not observed or the identity bridge is absent, do not fabricate the feature.

### id_fund_relay_nat_cnt (Payment-relay terminal natural-person count)
- description: Adapted from id_fund_relay_nat_cnt. Combine an initial natural-person expansion with two economically forward payment steps and a final natural-person binding.
- references: identity_fund_relay_derived_natural_person; dual_medium_alt_wechat_derivation
- paradigm: identity_fund_relay_derived_natural_person
- structure: Natural-person binding out/in; payer_side out to first recipients; payee_side in from those recipients to their recipients; natural-person binding out from terminal accounts.
- feature_form: expected_value_type=count; five relation steps, final count over distinct natural_person IDs. The incoming payee-side step is economically outgoing from the current payer set.
- discrimination: Compare against terminal account count, pure identity natural-person count, and reverse-relay variants. Stored direction and economic direction must not be conflated.
- example: Candidate name identity_payment_relay_terminal_natural_person_count. Logic enumerates the five hops and terminal deduplication.
- risk: The statistic describes the breadth of natural-person records reached after a payment relay and can expose identity dispersion not visible in a pure transfer view.
- boundary: It does not prove a fund loop or common control of terminal persons. Reversing the fourth hop or terminating at accounts creates another feature.

### layered_transfer_out_ratio (Second-layer payment amount divided by first-layer receipts)
- description: Rewritten from layered_transfer_out_ratio. Use monetary amounts on two different transfer layers to measure onward payment relative to initial receipts without introducing an identity prefix.
- references: pure_fund_chain_layered_transfer
- paradigm: pure_fund_chain_layered_transfer
- structure: Outgoing payee_side edges of the seed provide first-layer receipt amount; those upstream payers' outgoing payer_side edges provide the next payment layer. Additional transfer steps must be explicitly represented if used.
- feature_form: expected_value_type=ratio; numerator is second-layer payer-side total_amount, denominator is first-layer payee-side total_amount, both in fen; zero denominator yields 0.
- discrimination: A ratio can reduce scale effects, but ordinary businesses routinely reuse receipts for payments. Direction and layer selection matter more than a generic deep-chain label.
- example: Candidate name first_payer_onward_payment_to_seed_receipt_ratio. Logic identifies the two edge sets and their amounts independently before division.
- risk: The statistic provides an interpretable onward-payment scale measure that can be evaluated alongside time behavior and counterparty concentration.
- boundary: Never use vertex counts as a substitute for the stated monetary numerator and denominator. A pure account-count chain or an identity-anchored ratio belongs elsewhere.

### payer_fund_chain_merchant_amt (Multi-step personal payments ending at merchant payment)
- description: Adapted from payer_fund_chain_merchant_amt. Follow three explicit payer-side personal transfer steps and sum terminal payer-side merchant-payment edges.
- references: pure_fund_chain_layered_transfer; identity_anchored_controlled_wechat_admin_merchant_receive
- paradigm: pure_fund_chain_layered_transfer
- structure: personal_wechat_pay_transfer_payer_side out three times, then merchant_payment_receipt_payer_side out from terminal personal payers to merchant_pay_account while summing total_amount.
- feature_form: expected_value_type=amount; amount_unit=fen; final scalar sum over terminal merchant-payment edges, not the preceding personal-transfer amounts.
- discrimination: The exact number of personal relays and the terminal merchant relation distinguish this feature from direct merchant payment and reverse merchant-receipt tracing.
- example: Candidate name three_personal_payment_relay_merchant_amount_fen. Logic states that all three personal transfers use payer_side and only the fourth hop is merchant payment.
- risk: The measure connects layered personal payment neighborhoods to business-payment activity while preserving a clear monetary endpoint.
- boundary: Ordinary supplier and marketplace payments can form multi-step paths. No merchant administration or risk label is present; do not add one to the logic or plan.

### identity_anchored_merchant_flag (Administration existence after identity expansion)
- description: Rewritten from identity_anchored_merchant_flag with prohibited source risk labels removed. Test whether customer-associated or natural-person-associated accounts administer any merchant.
- references: identity_anchored_controlled_wechat_admin_merchant_receive
- paradigm: identity_anchored_controlled_wechat_admin_merchant_receive
- structure: Customer or natural-person binding out to the identity record, in to associated accounts, then personal_wechat_pay_account_is_merchant_super_administrator out to merchant_pay_account.
- feature_form: expected_value_type=flag; final exists over distinct merchant IDs, returning 1 for a nonempty endpoint set and 0 otherwise. Do not sum receipts or add a merchant-risk predicate to this flag.
- discrimination: Compare with direct administration and administered-merchant receipt amount. A structural existence flag can be sparse and should not inherit the calibration of a monetary feature.
- example: Candidate name identity_associated_merchant_administration_flag. Logic states the identity bridge and merchant endpoint without claiming merchant fraud.
- risk: The result isolates an administrator role as a reviewable structural association and leaves monetary activity to a complementary feature.
- boundary: Administration is not ownership or fraudulent behavior. The hypothesis is invalid when the identity bridge or administrator relation is absent.
