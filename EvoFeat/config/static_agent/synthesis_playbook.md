### relation_grounding (Relation and stored-direction grounding)
- when: Traversing any observed relation from one endpoint to another.
- do: Copy rel verbatim from the evidence chain and filter with has('relation', rel). From the stored src endpoint use outE().inV(); from the stored dst endpoint use inE().outV(). Keep payer_side and payee_side as distinct relation identifiers.
- example: To find the recipients recorded on a payer-side relation, traverse personal_wechat_pay_transfer_payer_side from its src endpoint with outE().has('relation', 'personal_wechat_pay_transfer_payer_side').inV().
- mechanism: The relation and stored direction jointly determine the measured endpoint set. Substituting a similar relation or reversing the edge changes the feature semantics.
- boundary: An intentionally direction-agnostic association may use bothE().has('relation', rel).otherV() only when logic explicitly requires it.

### exec_anon (Anonymous sub-traversals)
- when: A plan contains nested sub-traversals such as where(), union(), or(), and(), or not().
- do: Use anonymous traversals such as __.V() inside nested scopes and never restart with g.V(). Pass anonymous traversals to or(), and(), and not().
- example: Use where(__.V().hasId({{STEP:s1}}).outE().inV()) rather than starting a new full-graph traversal inside where().
- mechanism: A nested g.V() escapes the current traversal context and can cause syntax errors, full-graph scans, timeouts, or empty results.
- boundary: A single anchored plan without nested sub-traversals does not need this rule.

### amount_edge_const (Amount literals and units)
- when: Filtering or aggregating the edge property total_amount.
- do: Write total_amount and trade_count thresholds as integers in their stored units, such as has('total_amount', gt(10000)). Never rescale amount values.
- example: Use outE().has('relation', 'personal_wechat_pay_transfer_payer_side').has('total_amount', gt(5000)) for an observed 5000-fen threshold.
- mechanism: total_amount is an integer number of fen and trade_count is an integer. Quoted, fractional, or rescaled literals change the filter semantics.
- boundary: Count-only features that do not reference amount values do not need amount literals.

### query_form_placeholder (Unquoted execution placeholders)
- when: Using {{SEED}} or {{STEP:s1}} in a plan.
- do: Insert placeholders directly as hasId({{SEED}}) or hasId(within({{STEP:s1}})) without surrounding quotes; the executor supplies their values and types.
- example: Anchor with g.V().hasId({{SEED}}), then consume an upstream ID list with g.V().hasId(within({{STEP:s1}})).
- mechanism: Quoting turns the placeholder into a string literal and prevents correct ID-list substitution, often producing zero hits.
- boundary: Explicit literal IDs used outside the formal placeholder mechanism follow the graph's literal type rules.

### query_form_ratio (Python-side ratio reduction)
- when: The feature returns a ratio.
- do: Produce numerator and denominator in separate steps and use final.ratio with explicit scalar/count modes. Perform division in Python with deterministic zero-denominator protection; do not call asDouble() or divide inside Gremlin.
- example: Count the qualifying subset in one step and the full denominator set in another, then configure final as type=ratio with numerator_step and denominator_step.
- mechanism: The server does not provide the assumed floating-point conversion and division form. Separate scalar steps keep types explicit and make the reduction executable.
- boundary: Pure count, flag, or amount features do not use final.ratio.
