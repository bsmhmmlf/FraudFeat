### relation_grounding (Ground every relation and endpoint in the public schema)
- when: A feature hypothesis names a path, relation, or terminal entity that must become an executable traversal, especially after a relation-consistency mismatch.
- do: Match each required rel to relations[].en in the public schema, verify its stored src/dst entity types, copy it into has('relation', rel), and preserve the hypothesis's terminal object. For a within group, enumerate each observed rel rather than substituting a similar endpoint role.
- avoid: Inventing a missing relation, replacing a required endpoint with a convenient earlier one, reversing payer_side/payee_side to obtain a nonempty result, or silently truncating a path while claiming to implement the original feature.
- example: If the evidence chain specifies personal_wechat_pay_transfer_payee_side but the plan uses merchant_payment_receipt_payee_side, replace that predicate with outE().has('relation', 'personal_wechat_pay_transfer_payee_side').inV() while preserving the step's remaining computation.
- mechanism: Rewritten from real relation-consistency repairs: same-family relation substitutions and reversed endpoints produced queries that could execute but measured the wrong population. Relation and stored direction jointly define the evidence chain.
- boundary: A required relation absent from the public schema calls for an explicitly revised hypothesis before synthesis. An intentionally direction-agnostic association may use bothE().has('relation', rel).otherV() only when logic requires it.

### relation_public_direction (Separate stored direction from economic role)
- when: The path uses transfer, merchant receipt, withdrawal, or bank-card top-up relations.
- do: Determine which endpoint is the current node. From stored src use outE().inV(); from stored dst use inE().outV(). Check the economic payer/recipient meaning separately from the stored direction.
- avoid: Treating a path arrow or relation label as sufficient evidence of traversal direction, or describing an upstream step as an outgoing payment.
- example: personal_wechat_pay_transfer_payee_side is stored recipient-to-payer, so its outgoing traversal finds payers. wechat_pay_account_top_up_with_bank_card is stored account-to-card even though the card funds the account.
- mechanism: Correct syntax can still aggregate the wrong side when economic role and stored edge orientation are conflated.
- boundary: Mirror representations of an association must be normalized to the public relation and the corresponding inverse traversal, rather than copied as new relation types.

### depth_evidence_chain_coverage (Preserve all required main-path segments)
- when: A hypothesis contains multiple identity, transfer, merchant, or instrument segments.
- do: Map every main-path segment to a declared step and verify the final step's entity type. Distinguish an extra endpoint-existence filter from another main-path hop.
- avoid: Replacing the declared deep endpoint with an earlier count, skipping the identity segment after a transfer, or counting filter sub-traversals as path depth.
- example: An identity-transfer-identity account count requires the second subject-to-account expansion; counting the transfer counterparty alone implements a different feature.
- mechanism: A plausible numeric output does not prove that the intended evidence chain was traversed in full.
- boundary: Separate plans for numerator and denominator may share prefixes but must each cover their own required structure.

### query_form_seed_anchor (Anchor every computation to the seed or a derived set)
- when: A plan starts a graph traversal or creates a new computation branch.
- do: Begin with g_fraudfeat.V().hasId({{SEED}}) or with hasId(within({{STEP:sX}})) for a previously produced node set. State the branch's relation to the feature's seed.
- avoid: An unfiltered whole-graph aggregate that is unrelated to the seed, or a nested graph restart that changes the current element context.
- example: g_fraudfeat.V().hasId(within({{STEP:s2}})).outE().has('relation', 'personal_wechat_pay_transfer_payer_side') selects payments of the derived accounts.
- mechanism: Seed anchoring makes each value an account-level feature rather than a global statistic copied to every account.
- boundary: A separately defined global reference is not an account-local traversal and must be represented as a different input contract.

### query_form_step_dependency (Keep step references unique, ordered, and typed)
- when: A plan contains multiple steps or references {{SEED}} and {{STEP:sX}} placeholders.
- do: Give every step a unique ID, define referenced steps before use, declare depends_on and output, and insert placeholders without quotation marks. Use id_list sets for hasId(within(...)) inputs.
- avoid: Undefined or forward step references, quoted list placeholders, duplicate IDs, and interpreting a scalar amount as a node-ID set.
- example: A step with depends_on=['s1'] consumes hasId(within({{STEP:s1}})) and declares output='id_list' when it returns distinct node IDs.
- mechanism: Explicit dependencies and output types prevent accidental string substitution and make branch dataflow inspectable.
- boundary: final.ratio references numerator and denominator steps separately; both dependencies must be available before reduction.

### query_form_supported_flat_steps (Use the evaluator's implemented traversal subset)
- when: A hypothesis needs branching, endpoint qualification, or an aggregate that might tempt an unsupported Gremlin form.
- do: Prefer flat anchored steps using outE/inE, relation selection, inV/outV, dedup, id, and explicit value aggregation. Use the supported where(__.outE().has('relation', ...)) form for relation-existence filters.
- avoid: Assuming that every operation of a full Gremlin server is available in the Parquet evaluator, or hiding arithmetic in an unverified traversal extension.
- example: Two consecutive where(__.outE().has('relation', ...)) tests can require both customer and natural-person bindings at an account endpoint.
- mechanism: Constrained plans are easier to validate, interpret, and execute consistently across the framework and the standalone evaluator.
- boundary: A new query form should have an execution test before it becomes a reusable synthesis rule.

### query_form_ratio_modes_zero_guard (Define ratio operands and zero-denominator behavior)
- when: The declared expected_value_type is ratio.
- do: Compute numerator and denominator in separate steps and use final.type='ratio' with numerator_step, denominator_step, and explicit count or scalar modes. The shared executor returns 0 for a zero denominator and records the condition in the trace.
- avoid: Dividing inside Gremlin, silently changing the denominator population, or assuming that ratios of different sets are bounded by 1.
- example: Count reached natural_person records in the numerator and initial individual_customer records in the denominator, using count mode for both.
- mechanism: Explicit reductions preserve dimensional meaning and make undefined-denominator cases deterministic.
- boundary: A proportion uses a subset and its containing population; an expansion ratio can use distinct sets and exceed 1.

### aggregate_value_type_compat (Align output and final reduction with the measured quantity)
- when: Selecting a final reduction for a count, flag, amount, or ratio feature.
- do: Use count or exists on an ID list, scalar on an amount-sum step, and ratio on two declared operand steps. Keep the feature description, expected_value_type, and actual measurement consistent.
- avoid: Counting paths when the specification asks for entities, returning an amount for an existence flag, or labeling every scalar result as a count.
- example: A bank-card endpoint list followed by final.exists is a binding flag; the sum of withdrawal-edge total_amount followed by final.scalar is a withdrawal amount.
- mechanism: The same path can support different statistics, but the terminal reduction determines which business question is answered.
- boundary: A reduction change creates a different feature even when its graph prefix is unchanged.

### aggregate_distinct_target_units (Deduplicate the correct entities or edges)
- when: Multiple paths reach the same account, identity record, merchant, or contributing transaction edge.
- do: Deduplicate endpoints before an entity count and deduplicate contributing edges before an amount sum. Keep revisited accounts unless the hypothesis explicitly excludes them.
- avoid: Using path multiplicity as an entity count, multiplying an edge amount by the number of identity routes to it, or silently excluding the seed and earlier layers.
- example: dedup().id().toList() creates an entity set; outE().has('relation', relation_name).dedup().values('total_amount').sum() aggregates distinct selected edges.
- mechanism: Entity and edge deduplication answer different questions and prevent repeated traversal routes from inflating the value.
- boundary: Runtime node-list limits affect coverage. A deduplicated set may still be truncated by the configured execution contract.

### amount_edge_fen_and_trade_count (Aggregate stored edge amounts in RMB fen)
- when: Filtering or aggregating total_amount or trade_count on transaction relations.
- do: Keep total_amount as integer RMB fen and trade_count as an integer count. Select transaction edges before reading these properties; sum total_amount on those edges and label the result with its stored unit.
- avoid: Reading an edge amount after moving to a vertex, converting fen to yuan inside the plan, or interpreting an aggregated edge as an individual transaction log.
- example: outE().has('relation', 'wechat_pay_withdrawal_to_bank_card').dedup().values('total_amount').sum() yields withdrawal amount in fen; 100 fen equals 1 yuan.
- mechanism: Unit preservation and correct property scope make monetary plans comparable across the generator and the evaluator.
- boundary: Human-facing presentation may state the unit explicitly, while executable feature values retain the Parquet schema's stored units.
