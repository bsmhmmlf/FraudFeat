"""Endpoint-contract and obsolete relation-query regression tests."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from fraudfeat.direct_iv import _relation_endpoints
from fraudfeat.diversity import SchemaIndex, build_graph
from fraudfeat.graph_store import GraphStore, load_schema, relation_endpoints
from fraudfeat.gremlin_engine import EngineError, execute_plan_local, validate_plan
from fraudfeat.interpretability import build_domain_brief

from .test_fraudfeat import C, CUST_REL, REL, W, _plan, _write_graph


class RelationSchemaTest(unittest.TestCase):
    def test_direct_endpoint_roles(self):
        relation = {"en": CUST_REL, "src": W, "dst": C}
        self.assertEqual(relation_endpoints(relation), (W, C))
        self.assertEqual(_relation_endpoints(relation, W), (W, C, ["src"]))
        self.assertEqual(_relation_endpoints(relation, C), (W, C, ["dst"]))
        self.assertEqual(_relation_endpoints({"en": REL, "src": W, "dst": W}, W),
                         (W, W, ["src", "dst"]))

    def test_legacy_schema_is_rejected_without_modification(self):
        for mixed in (False, True):
            relation = {"en": CUST_REL, "join_fields": [{"src": W, "dst": C}]}
            if mixed:
                relation.update(src=W, dst=C)
            schema = {"entities": [{"entity_type": W}, {"entity_type": C}],
                      "relations": [relation]}
            original = copy.deepcopy(schema)
            for consumer in (SchemaIndex, build_domain_brief):
                with self.subTest(mixed=mixed, consumer=consumer.__name__):
                    with self.assertRaisesRegex(ValueError, "obsolete join_fields"):
                        consumer(schema)
            with self.assertRaisesRegex(ValueError, "obsolete join_fields"):
                _relation_endpoints(relation, W)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "schema.json"
                path.write_text(json.dumps(schema), encoding="utf-8")
                before = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "obsolete join_fields"):
                    load_schema(tmp)
                self.assertEqual(path.read_bytes(), before)
            self.assertEqual(schema, original)

    def test_missing_or_invalid_endpoint_is_rejected(self):
        for bad in (None, "", " ", [], 123):
            for side in ("src", "dst"):
                relation = {"en": CUST_REL, "src": W, "dst": C, side: bad}
                with self.subTest(side=side, value=bad):
                    with self.assertRaisesRegex(ValueError, "nonempty src and dst"):
                        relation_endpoints(relation)

    def test_schema_requires_no_version_or_format(self):
        schema = {"entities": [{"entity_type": W}, {"entity_type": C}],
                  "relations": [{"en": CUST_REL, "src": W, "dst": C}]}
        self.assertEqual(SchemaIndex(schema).rel_meta(CUST_REL), (CUST_REL, W, C))
        self.assertIn(f"{W} -> {C}", build_domain_brief(schema))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schema.json"
            path.write_text(json.dumps(schema), encoding="utf-8")
            self.assertEqual(load_schema(tmp), schema)


class RelationQueryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = _write_graph(Path(self.temp.name))
        self.store = GraphStore(root)

    def test_legacy_property_queries_fail_even_on_empty_traversals(self):
        suffixes = [
            f".has('join_field','{REL}').inV().id().toList()",
            f'.has("join_field",within("{REL}")).inV().id().toList()',
            f".has('relation','join_field','{REL}').inV().id().toList()",
            ".has('join_field').inV().id().toList()",
            ".hasNot('join_field').inV().id().toList()",
            ".values('join_field').toList()",
            ".values('relation','join_field').toList()",
            ".properties('join_field').toList()",
            ".valueMap('join_field').toList()",
            ".propertyMap('join_field').toList()",
            ".elementMap('relation','join_field').toList()",
            ".order().by('join_field').inV().id().toList()",
            ".order().by(values('join_field')).inV().id().toList()",
            f".or(__.has('join_field','{REL}'),__.has('relation','{REL}')).inV().id().toList()",
            f".where(__.has('join_field','{REL}')).inV().id().toList()",
            f".has(join_field,'{REL}').inV().id().toList()",
        ]
        for property_name in ("join_field", "join_fields", "jf"):
            for template in suffixes:
                suffix = template.replace("join_field", property_name)
                plan = _plan("g.V().hasId({{SEED}}).outE()" + suffix)
                dependent_plan = {
                    "steps": [
                        {"id": "s1", "output": "id_list",
                         "gremlin": "g.V().hasId({{SEED}}).has('state','absent').id().toList()"},
                        {"id": "s2", "output": "id_list", "depends_on": ["s1"],
                         "gremlin": "g.V().hasId({{STEP:s1}}).outE()" + suffix}],
                    "final": {"type": "count", "step": "s2"},
                }
                for candidate in (plan, dependent_plan):
                    with self.subTest(query=suffix, steps=len(candidate["steps"])):
                        valid, error = validate_plan(candidate)
                        self.assertFalse(valid)
                        self.assertIn("Obsolete relation property", error)
                        for seed in ("A00000001", "missing"):
                            value, _trace, error = execute_plan_local(self.store, candidate, seed)
                            self.assertIsNone(value)
                            self.assertIn("Obsolete relation property", error)

    def test_legacy_text_as_value_is_not_a_property(self):
        for value_text in ("join_field", "join_fields", "jf"):
            with self.subTest(value=value_text):
                plan = _plan(f"g.V().hasId({{{{SEED}}}}).has('state','{value_text}').id().toList()")
                self.assertEqual(validate_plan(plan), (True, ""))
                value, _trace, error = execute_plan_local(self.store, plan, "A00000001")
                self.assertEqual((value, error), (0, ""))

    def test_diversity_rejects_legacy_queries(self):
        for property_name in ("join_field", "join_fields", "jf"):
            with self.subTest(property=property_name):
                plan = _plan(f"g.V().hasId({{{{SEED}}}}).outE().has('{property_name}','{REL}')"
                             ".inV().id().toList()")
                with self.assertRaisesRegex(EngineError, "Obsolete relation property"):
                    build_graph({"gremlin_plan": plan}, SchemaIndex(self.store.schema))


if __name__ == "__main__":
    unittest.main()
