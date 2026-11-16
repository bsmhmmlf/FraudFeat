### payee_bind_wechat_count_bias (Receipt-first identity-bridge count overestimation)
- applicable_family: Three-hop features begin with outgoing personal_wechat_pay_transfer_payee_side from the seed recipient, cross one customer or natural-person binding, reverse it to accounts, and end with a count.
- true_distribution: Four real observations covered IV 0.045-0.196. Earlier anchors were 0.045 and 0.045, with later observations 0.196.
- bias: Predictions 0.300 -> 0.045, 0.300 -> 0.045, and 0.380 -> 0.196 overestimated by 0.255, 0.255, and 0.184. A later 0.160 -> 0.196 underestimated by 0.036, showing partial correction rather than a permanent direction.
- criteria: Do not equate a reachable deep identity path with strong discrimination. The source errors ignored limited black-sample hit rates and ordinary shallow identity associations.
- action: For an exact structural match, start from IV 0.05-0.20 with confidence about 0.60-0.65, then adjust using comparable measured examples. Do not force every prediction downward after one reversed residual.
- avoid: Do not apply these numbers to payer-side starts, amount or flag reductions, or deeper five-hop identity expansions.
- boundary: Evidence comprises four observations. Reopen the interval when new measured examples repeatedly reverse the bias.

### transfer_initiated_customer_relay_count_underest (Transfer-first customer-relay count correction)
- applicable_family: A payer-side or payee-side transfer starts the path, a second transfer hop follows, and a customer binding then expands to associated accounts before the terminal account count.
- true_distribution: Four observations were 0.634, 1.763, 0.634, and 0.634, giving a measured range of 0.634-1.763 rather than a single high anchor.
- bias: Predictions 0.170 -> 0.634, 0.230 -> 1.763, and 0.170 -> 0.634 underestimated by 0.464, 1.533, and 0.464. A later 1.400 -> 0.634 overestimated by 0.766 after excessive upward correction.
- criteria: Recognize the transfer-before-identity topology, but do not assign the highest family value to every instance. Endpoint, exact relay direction, and comparable cluster evidence must be checked.
- action: Use 0.6-0.9 for examples comparable with the 0.634 observations and 1.2-1.8 only when the example set supports the 1.763 subpopulation. Keep confidence near 0.60-0.65 and state the uncertainty.
- avoid: Do not use this entry for identity-first paths, three-hop direct identity counts, terminal customer counts, or amount/ratio/flag reductions.
- boundary: The apparent low/high subgroups are historical evidence, not a deterministic rule based on one subgraph-size threshold.

### transfer_initiated_fund_relay_end_customer_count_bias (Transfer-relay terminal-customer count overestimation)
- applicable_family: Four-to-five-hop paths begin with payer-side transfer, continue through two or three personal transfer steps, and terminate at distinct individual_customer records rather than accounts.
- true_distribution: The two available real observations were IV 0.125 and 0.411.
- bias: Predictions 0.600 -> 0.125 and 0.620 -> 0.411 overestimated by 0.475 and 0.209. Both errors came from analogy with terminal-account relay counts.
- criteria: Changing the terminal from accounts to customer records changes the normal-sample mechanism. Do not inherit the higher terminal-account calibration merely because the transfer prefix matches.
- action: Start from IV 0.10-0.45 with confidence about 0.55-0.60 until more exact terminal-customer evidence is available.
- avoid: Do not apply to identity-first paths, payee-side-only starts outside the observed family, terminal account counts, or non-count reductions.
- boundary: Only two same-window observations support this interval; confidence should remain limited and the entry should be revised with new measurements.

### customer_bound_wechat_amount_underest (Customer-associated receipt-amount bias reversal)
- applicable_family: Three-hop amount features use wechat_pay_account_bind_individual_customer out/in to associated accounts and then aggregate a personal transfer amount at the terminal hop.
- true_distribution: Four measured examples were 0.699 and a later example was 0.297, producing an observed range of 0.297-0.699.
- bias: Predictions 0.650 -> 0.699, 0.360 -> 0.699, 0.698 -> 0.699, and 0.698 -> 0.699 mostly underestimated; the later 0.699 -> 0.297 overestimated by 0.402.
- criteria: Do not lower IV because of imprecise direction wording, but also do not copy 0.699 to every same-prefix amount feature. Actual terminal relation and amount semantics control the analogy.
- action: Use the broad observed 0.30-0.70 range as a prior, with confidence about 0.65-0.70, and narrow it only with exact terminal-edge and direction matches.
- avoid: Do not transfer the interval to count/flag outputs, natural-person-first paths, payer-side-only variants, or deeper identity chains.
- boundary: The family contains both historical underestimation and a substantial later overestimate; a one-direction additive correction is inappropriate.

### identity_multi_hop_wechat_underest (Pure-identity endpoint-specific calibration)
- applicable_family: Three-to-five-hop paths contain only customer, natural-person, account, or public merchant-administration relations and no personal-transfer, merchant-payment, withdrawal, or recharge edge. Terminal count, ratio, and flag forms must be separated.
- true_distribution: Real observations included unfiltered terminal-account counts mostly 1.079-1.221 with one 6.414 outlier, terminal-customer counts 0.417-0.474, terminal-natural-person counts 0.382-0.429, and identity ratios 1.240-1.516.
- bias: The evaluator both overestimated 6.000 -> 1.221 and 3.800 -> 1.079 and underestimated 1.150 -> 6.414. Customer-count 0.430 -> 0.474 was only a slight underestimate. The dominant lesson is endpoint confusion, not universal underestimation.
- criteria: Match terminal entity, hop sequence, and filters before matching the family name. An extreme account-count observation must not become the default for all identity paths.
- action: Use 0.3-0.5 for terminal natural-person counts, 0.40-0.48 for terminal customer counts, and 1.0-1.5 for comparable identity ratios. For unfiltered terminal-account counts, prioritize the 1.0-2.0 central range and require strong exact-match evidence before approaching the 6.414 outlier.
- avoid: Do not apply these intervals to fund-bearing paths, two-hop bindings, amount sums, or label-dependent merchant flags.
- boundary: The evidence pool mixes repeated revisions and an extreme value; preserve confidence near 0.60-0.65 and disclose which endpoint subinterval is being used.

### nat_anchor_payer_relay_count_underest (Natural-person anchored payment-relay count bias)
- applicable_family: Five-hop paths start with wechat_pay_account_bind_natural_person, reverse to associated accounts, continue through payer-side personal transfers, and count terminal accounts.
- true_distribution: Four real observations covered 0.802-1.070; unfiltered historical values were 0.802, 0.892, and 0.939, while a filtered example reached 1.070.
- bias: Predictions included 0.350 -> 0.892 underestimation, 1.450 -> 0.939 and 1.420 -> 1.070 overestimation, and 0.950 -> 0.802 overestimation. Three of four observations were overestimated.
- criteria: Avoid both the early overly conservative prior and later copying of very high same-family memories. Explicitly distinguish filtered from unfiltered terminal counts.
- action: For unfiltered exact matches, start at 0.8-1.0. The observed filtered example supports about 1.0-1.2 only for the same filter semantics. Keep confidence near 0.60-0.65.
- avoid: Do not use for customer-first paths, payee-side relays, amount/flag outputs, terminal natural-person counts, or four-hop variants.
- boundary: Four observations are enough for a cautious interval, not enough to assume that adding a filter always raises IV.

### identity_anchored_fund_relay_fanout_bias (Identity-anchored transfer-family calibration ranges)
- applicable_family: Three-to-five-hop features begin with customer or natural-person identity expansion and later include personal transfer, merchant payment, or withdrawal. Count, ratio, and amount forms require separate calibration.
- true_distribution: Real subfamilies ranged from 0.020 to 1.990 overall. Reverse-then-forward account counts were 0.787-1.990, unfiltered payer fan-out 0.870-1.751, payee tracing 0.710-0.928, and withdrawal amounts 0.294-0.666 in the cited examples.
- bias: Earlier examples were mostly underestimated, but later predictions reversed: about 1.95 -> 0.787 for reverse-then-forward count, 1.65 -> 0.870 for payer fan-out, and 1.10 -> 0.957 for terminal customer count.
- criteria: Do not assign a single identity-plus-transfer IV. Compare relay order, stored direction, terminal entity, reduction, and filters, and use multiple windows rather than one very high retrieved memory.
- action: Use 0.8-2.0 for reverse-then-forward account counts, 0.8-1.8 for unfiltered payer fan-out, 0.7-1.0 for pure payee tracing, and 0.25-0.80 for comparable withdrawal amounts. Treat customer and natural-person terminal counts and ratios as their own subfamilies.
- avoid: Do not apply to pure identity walks, transfer-first paths, the specialized natural-person five-hop count entry, or two-hop direct associations.
- boundary: These are broad historical priors with mixed residuals. Confidence around 0.60-0.70 is appropriate only when the exact subfamily is identified.

### payer_initiated_nat_bound_pay_amount_bias (Payer-first natural-person payment amount bias)
- applicable_family: Four-hop amount features start with payer-side personal transfer, cross a natural-person binding to associated accounts, and sum a later payer-side transfer amount.
- true_distribution: Two matching real observations both had IV 0.195.
- bias: The historical prediction 0.330 -> 0.195 overestimated by 0.135; a later 0.180 -> 0.195 differed by only 0.015 after calibration.
- criteria: Do not transfer the high prior of deep identity-anchored fan-out to this shorter transfer-first monetary path. The natural-person bridge and terminal amount are essential.
- action: Start from IV 0.15-0.25 with confidence about 0.55-0.60, while recording that only two measured examples support the interval.
- avoid: Do not use for identity-first paths, payee-side starts, terminal counts or flags, direct customer endpoints, or three-hop variants.
- boundary: The later accurate prediction demonstrates successful calibration; it does not justify continually decreasing the prior.

### pure_fund_chain_layered_transfer_count_bias (Transfer-only count calibration by depth and direction)
- applicable_family: Three-to-five-hop account counts contain only payer-side and payee-side personal transfer relations, with no identity, merchant-administration, or instrument relation.
- true_distribution: Four real observations were 0.347 for a three-hop example, 0.136 for a four-hop payer/mixed example, 0.415 for a five-hop example, and 0.516 for a four-hop payee-tracing example.
- bias: Predictions were 0.420 -> 0.136, 0.170 -> 0.347, 0.430 -> 0.415, and 0.220 -> 0.516. Both overestimation and underestimation occurred, so hop count alone is insufficient.
- criteria: Match the exact relation-direction sequence. In particular, do not reuse a five-hop value for a four-hop mixed path or a low payee-bridge identity-family value for pure four-hop payee tracing.
- action: Use 0.3-0.4 for comparable three-hop paths, 0.1-0.3 for four-hop payer/mixed paths, 0.3-0.5 for four-hop payee tracing, and about 0.4-0.5 for the observed five-hop form. Keep confidence near 0.55-0.65 by subgroup.
- avoid: Do not apply to identity-containing paths, amount or ratio outputs, terminal customer/natural-person counts, or two-hop direct transfers.
- boundary: Each directional subgroup has very few observations; the intervals are starting points and must not be interpreted as monotonic depth effects.

### pure_fund_chain_to_merchant_amount_bias (Pure-transfer merchant-payment amount calibration)
- applicable_family: Four-hop amount features begin with payer-side personal transfer, continue through two payee-side reverse-tracing steps, and sum terminal merchant_payment_receipt_payer_side amounts without identity mediation.
- true_distribution: The two real observations were IV 0.184 and 0.207.
- bias: Predictions 0.120 -> 0.207 and 0.210 -> 0.184 had opposite residuals of -0.087 and +0.026. Both measurements nevertheless remained in a narrow low-IV range.
- criteria: A retrieved same-family memory can stabilize the estimate, but ordinary merchant payments are common and must not inherit a high identity-anchored fan-out prior.
- action: Start from IV 0.15-0.25 with confidence about 0.55-0.60; do not apply a systematic upward or downward correction from these mixed residuals.
- avoid: Do not use for identity-first paths, merchant-administration flags, terminal account counts, or paths with a different relay-direction sequence.
- boundary: Only two observations support this entry. Preserve the exact four-hop and terminal merchant-payment requirements.

### pure_fund_chain_merchant_flag_overest (Transfer-chain merchant flag label boundary)
- applicable_family: The historical source entry concerned a transfer-first path ending at a merchant flag conditioned on an internal risk label. That label is deliberately absent from the public FraudFeat schema.
- true_distribution: The label-conditioned source observations were IV 0.020, 0.046, and 0.070. They are retained only as provenance for the original overestimation diagnosis, not as targets for a different public feature.
- bias: Source predictions 0.230 -> 0.070, 0.080 -> 0.046, and 0.120 -> 0.020 overestimated by 0.160, 0.034, and 0.100. Removing the label would change the feature and invalidate these numerical pairs.
- criteria: Never reconstruct the source predicate from a public merchant attribute or treat administration alone as a risk label. Distinguish the historical label-conditioned flag from label-free merchant existence.
- action: Do not emit the historical feature or reuse its numeric IV interval for a public label-free flag. Suppress the label-dependent candidate and evaluate any administration-only or transfer-terminal existence feature using its own measured examples.
- avoid: Do not claim that ordinary merchants reached through payments are fraudulent, and do not transfer calibration from identity-anchored administration flags or merchant amount features.
- boundary: This entry is a compatibility and anti-leakage lesson. It provides no direct numerical calibration for a valid public-schema feature until matching label-free measurements exist.
