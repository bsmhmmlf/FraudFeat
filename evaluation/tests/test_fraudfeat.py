# -*- coding: utf-8 -*-
"""Standalone offline tests for the FraudFeat evaluation package.

Run with::

    python -m unittest discover -s tests -t . -v
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from fraudfeat.autobin import AutobinSplit
from fraudfeat.direct_iv import main as direct_main
from fraudfeat.diversity import SchemaIndex, batch_metrics, build_graph, stat_signature
from fraudfeat.effectiveness import (
    evaluate_feature, load_eval_set, main as eff_main, summarize as effectiveness_summary,
)
from fraudfeat.embedding_diversity import (
    _embed_openai, cosine_matrix, embed_texts, feature_text, main as embedding_main,
)
from fraudfeat.graph_store import GraphStore
from fraudfeat.gremlin_engine import execute_plan_local, validate_plan
from fraudfeat.interpretability import (
    RUBRIC_PROMPT, build_domain_brief, build_requests, main as interpret_main,
    rubric_sha1, score_requests, summarize as interpretability_summary,
)
from fraudfeat.pool import prepare_features, ranked_indices
from fraudfeat.sql_effectiveness import (
    connect_release, execute_sql_query, evaluate_feature_sql,
    main as sql_main, validate_sql,
)


W = "wechat_pay_account"
C = "individual_customer"
REL = "personal_wechat_pay_transfer_payer_side"
CUST_REL = "wechat_pay_account_bind_individual_customer"


def _write_graph(root: Path) -> Path:
    (root / "nodes").mkdir(parents=True, exist_ok=True)
    (root / "edges").mkdir(parents=True, exist_ok=True)
    (root / "nodes" / f"{W}.csv").write_text(
        "unique_id,entity_type,reg_time,state\n"
        f"A00000001,{W},20220101000000,Deregistered\n"
        f"A00000002,{W},20240101000000,Registered\n"
        f"A00000003,{W},0,Registered\n", encoding="utf-8")
    (root / "nodes" / f"{C}.csv").write_text(
        "unique_id,entity_type,cre_type\nS00000001,%s,Identity card\n" % C,
        encoding="utf-8")
    (root / "edges" / "rel_00.csv").write_text(
        "src,dst,relation,total_amount,trade_count,top_k\n"
        f"A00000001,A00000002,{REL},10000,5,1\n"
        f"A00000001,A00000003,{REL},4000,2,2\n"
        f"A00000002,A00000003,{REL},20000,7,1\n", encoding="utf-8")
    (root / "edges" / "rel_01.csv").write_text(
        "src,dst,relation\n"
        f"A00000001,S00000001,{CUST_REL}\n", encoding="utf-8")
    (root / "schema.json").write_text(json.dumps({
        "version": 4, "format": "csv",
        "entities": [
            {"entity_type": W, "description": "A personal WeChat Pay account.",
             "file": f"nodes/{W}.csv",
             "fields": [{"name": "reg_time"}, {"name": "state"}]},
            {"entity_type": C, "description": "An individual customer record.",
             "file": f"nodes/{C}.csv", "fields": [{"name": "cre_type"}]}],
        "relations": [
            {"index": 0, "en": REL, "is_trade": True,
             "join_fields": [{"src": W, "dst": W}],
             "file": "edges/rel_00.csv"},
            {"index": 1, "en": CUST_REL, "is_trade": False,
             "join_fields": [{"src": W, "dst": C}],
             "file": "edges/rel_01.csv"}],
        "labels": {"file": "labels.csv"}}, ensure_ascii=False), encoding="utf-8")
    (root / "labels.csv").write_text(
        "unique_id,label\nA00000001,1\nA00000002,0\nA00000003,0\n", encoding="utf-8")
    return root


def _plan(gremlin: str, out: str = "id_list", final=None) -> dict:
    final = final or ({"type": "count", "step": "s1"} if out == "id_list"
                      else {"type": "scalar", "step": "s1"})
    return {"steps": [{"id": "s1", "output": out, "gremlin": gremlin}], "final": final}


def _write_public_parquet_graph(root: Path) -> Path:
    """Build a small release-shaped Parquet graph with public relations and English enums."""
    (root / "nodes").mkdir(parents=True)
    (root / "edges").mkdir(parents=True)
    bind_account = "wechat_pay_account_bind_natural_person"
    bind_customer = "natural_person_bind_individual_customer"
    schema = {
        "schema_version": "4", "format": "parquet",
        "entities": [
            {"entity_type": W, "description": "A personal WeChat Pay account.",
             "file": "nodes/wechat.parquet",
             "fields": [{"name": "state", "dtype": "string",
                         "values": {"Registered": "Registered", "Deregistered": "Deregistered"}}]},
            {"entity_type": "natural_person", "description": "A natural-person identity.",
             "file": "nodes/natural.parquet",
             "fields": [{"name": "gender", "dtype": "string",
                         "values": {"Male": "Male", "Female": "Female"}},
                        {"name": "birthday", "dtype": "string"}]},
            {"entity_type": C, "description": "An individual customer record.",
             "file": "nodes/customer.parquet",
             "fields": [{"name": "last_dereg_time", "dtype": "string"}]},
        ],
        "relations": [
            {"index": 0, "en": REL, "is_trade": True, "file": "edges/transfers.parquet",
             "join_fields": [{"src": W, "dst": W}],
             "description": "top_k is the payee's rank among the payer's top eight payees."},
            {"index": 1, "en": bind_account, "is_trade": False, "file": "edges/account.parquet",
             "join_fields": [{"src": W, "dst": "natural_person"}]},
            {"index": 2, "en": bind_customer, "is_trade": False, "file": "edges/customer.parquet",
             "join_fields": [{"src": "natural_person", "dst": C}]},
        ],
        "labels": {"file": "labels.parquet"},
    }
    (root / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    pd.DataFrame([
        {"unique_id": "A1", "entity_type": W,
         "state": "Registered", "dereg_time": "0"},
        {"unique_id": "A2", "entity_type": W,
         "state": "Registered", "dereg_time": "0"},
        {"unique_id": "A3", "entity_type": W,
         "state": "Deregistered", "dereg_time": "2020-01-01 00:00:00"},
        {"unique_id": "A4", "entity_type": W,
         "state": "Deregistered", "dereg_time": "2021-01-01 00:00:00"},
    ]).to_parquet(root / "nodes/wechat.parquet", index=False)
    pd.DataFrame([
        {"unique_id": "N1", "entity_type": "natural_person", "gender": "Male",
         "birthday": "1990-01-01"},
        {"unique_id": "N2", "entity_type": "natural_person", "gender": "Female",
         "birthday": "1985-05-05"},
    ]).to_parquet(root / "nodes/natural.parquet", index=False)
    pd.DataFrame([{"unique_id": "C1", "entity_type": C,
                   "last_dereg_time": "2022-01-01 00:00:00"}]).to_parquet(
        root / "nodes/customer.parquet", index=False)
    pd.DataFrame([
        {"src": "A1", "dst": "A2", "relation": REL, "total_amount": 1000, "trade_count": 2, "top_k": 1},
        {"src": "A1", "dst": "A3", "relation": REL, "total_amount": 500, "trade_count": 1, "top_k": 2},
        {"src": "A2", "dst": "A3", "relation": REL, "total_amount": 800, "trade_count": 1, "top_k": 1},
        {"src": "A4", "dst": "A1", "relation": REL, "total_amount": 300, "trade_count": 1, "top_k": 1},
    ]).to_parquet(root / "edges/transfers.parquet", index=False)
    pd.DataFrame([
        {"src": "A1", "dst": "N1", "relation": bind_account},
        {"src": "A2", "dst": "N1", "relation": bind_account},
        {"src": "A3", "dst": "N2", "relation": bind_account},
        {"src": "A4", "dst": "N2", "relation": bind_account},
    ]).to_parquet(root / "edges/account.parquet", index=False)
    pd.DataFrame([
        {"src": "N1", "dst": "C1", "relation": bind_customer},
        {"src": "N2", "dst": "C1", "relation": bind_customer},
    ]).to_parquet(root / "edges/customer.parquet", index=False)
    pd.DataFrame({"unique_id": ["A1", "A2", "A3", "A4"], "label": [1, 0, 0, 1]}).to_parquet(
        root / "labels.parquet", index=False)
    features = [
        {"feature_name": "transfer_out_degree", "display_name": "Outgoing transfer counterparties",
         "logic": "Count distinct outgoing transfer counterparties", "expected_value_type": "count",
         "explanation": "More transfer counterparties can indicate broader account connectivity.",
         "gremlin_plan": _plan(f"g.V().hasId({{{{SEED}}}}).outE().has('relation','{REL}')"
                               ".inV().dedup().id().toList()")},
        {"feature_name": "linked_identities", "display_name": "Linked individual customers",
         "logic": "Count individual customers linked through a natural person",
         "expected_value_type": "count", "explanation": "Shared identity links can connect accounts.",
         "gremlin_plan": _plan(
             f"g.V().hasId({{{{SEED}}}}).outE().has('relation','{bind_account}').inV()"
             f".outE().has('relation','{bind_customer}').inV().dedup().id().toList()")},
    ]
    (root / "features.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in features), encoding="utf-8")
    return root


class EngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = _write_graph(Path(cls._tmp.name))
        cls.store = GraphStore(cls.root, load_labels=False)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_one_hop_and_amount(self) -> None:
        hop = (f"g.V().hasId({{{{SEED}}}}).outE().has('relation','{REL}')"
               ".inV().dedup().id().toList()")
        self.assertEqual(execute_plan_local(self.store, _plan(hop), "A00000001")[0], 2)
        self.assertEqual(execute_plan_local(self.store, _plan(hop), "A00000003")[0], 0)

        amt = (f"g.V().hasId({{{{SEED}}}}).outE().has('relation','{REL}')"
               ".has('total_amount', gt(50.0)).inV().dedup().id().toList()")
        self.assertEqual(execute_plan_local(self.store, _plan(amt), "A00000001")[0], 2)
        self.assertEqual(execute_plan_local(self.store, _plan(amt), "A00000001",
                                            amount_unit="yuan")[0], 1)

        sm = (f"g.V().hasId({{{{SEED}}}}).outE().has('relation','{REL}')"
              ".values('total_amount').sum()")
        val, _t, err = execute_plan_local(self.store, _plan(sm, out="scalar"),
                                          "A00000001")
        self.assertEqual(err, "")
        self.assertAlmostEqual(val, 14000.0, places=9)
        yuan, _t, err = execute_plan_local(self.store, _plan(sm, out="scalar"),
                                           "A00000001", amount_unit="yuan")
        self.assertEqual(err, "")
        self.assertAlmostEqual(yuan, 140.0, places=9)

    def test_vertex_props_are_strings(self) -> None:
        quoted = "g.V().hasId({{SEED}}).has('state','Deregistered').dedup().id().toList()"
        self.assertEqual(execute_plan_local(self.store, _plan(quoted), "A00000001")[0], 1)
        numeric = "g.V().hasId({{SEED}}).has('state',2).dedup().id().toList()"
        value, trace, err = execute_plan_local(self.store, _plan(numeric), "A00000001")
        self.assertEqual(err, "")
        self.assertEqual(value, 0)                  # Numeric literal cannot match a string property.
        self.assertTrue(any(t.get("warnings") for t in trace))

    def test_validate_requires_seed(self) -> None:
        ok, msg = validate_plan({"steps": [{"id": "s1", "output": "id_list",
                                            "gremlin": "g.V().outE().id().toList()"}],
                                 "final": {"type": "count", "step": "s1"}})
        self.assertFalse(ok)
        self.assertIn("anchor", msg)

    def test_closed_set_operator_errors(self) -> None:
        plan = _plan("g.V().hasId({{SEED}}).repeat(out()).times(3).id().toList()")
        value, _t, err = execute_plan_local(self.store, plan, "A00000001")
        self.assertIsNone(value)
        self.assertIn("outside the closed operator set", err)

    def test_pred_namespace_prefix(self) -> None:
        plain = ("g.V().hasId({{SEED}}).has('reg_time', gt('2023-01-01'))"
                 ".dedup().id().toList()")
        ns = ("g.V().hasId({{SEED}}).has('reg_time', P.gt('2023-01-01'))"
              ".dedup().id().toList()")
        self.assertEqual(execute_plan_local(self.store, _plan(plain), "A00000002")[0], 1)
        self.assertEqual(execute_plan_local(self.store, _plan(ns), "A00000002")[0], 1)
        value, _t, err = execute_plan_local(
            self.store, _plan(plain.replace("gt(", "P.foo(")), "A00000002")
        self.assertIsNone(value)
        self.assertIn("outside the closed predicate set", err)

    def test_nested_predicate_rejected(self) -> None:
        plan = _plan("g.V().hasId({{SEED}}).is(P.neq(within(['A00000002']))).id().toList()")
        value, _t, err = execute_plan_local(self.store, plan, "A00000001")
        self.assertIsNone(value)
        self.assertIn("Nested predicate", err)

    def test_is_on_vertex_set(self) -> None:
        head = (f"g.V().hasId({{{{SEED}}}}).outE().has('relation','{REL}')"
                ".inV().dedup()")
        plan = _plan(head + ".is(within(['A00000002','A00000003'])).id().toList()")
        self.assertEqual(execute_plan_local(self.store, plan, "A00000001")[0], 2)
        plan2 = _plan(head + ".is(without(['A00000002'])).id().toList()")
        self.assertEqual(execute_plan_local(self.store, plan2, "A00000001")[0], 1)
        plan_edge = _plan(f"g.V().hasId({{{{SEED}}}}).outE().has('relation','{REL}')"
                          ".is('x').id().toList()")
        value, _t, err = execute_plan_local(self.store, plan_edge, "A00000001")
        self.assertIsNone(value)
        self.assertIn("edge sets", err)

    def test_has_single_arg_existence(self) -> None:
        head = (f"g.V().hasId({{{{SEED}}}}).outE().has('relation','{CUST_REL}')"
                ".inV()")
        self.assertEqual(
            execute_plan_local(self.store, _plan(head + ".has('cre_type').id().toList()"),
                               "A00000001")[0], 1)
        self.assertEqual(
            execute_plan_local(self.store, _plan(head + ".has('state').id().toList()"),
                               "A00000001")[0], 0)

    def test_nontrade_has_no_implicit_mirror(self) -> None:
        outgoing = (f"g.V().hasId({{{{SEED}}}}).outE().has('relation','{CUST_REL}')"
                    ".inV().dedup().id().toList()")
        incoming = (f"g.V().hasId({{{{SEED}}}}).inE().has('relation','{CUST_REL}')"
                    ".outV().dedup().id().toList()")
        self.assertEqual(execute_plan_local(self.store, _plan(outgoing), "A00000001")[0], 1)
        self.assertEqual(execute_plan_local(self.store, _plan(incoming), "A00000001")[0], 0)

    def test_count_local(self) -> None:
        g = (f"g.V().hasId({{{{SEED}}}}).outE().has('relation','{REL}')"
             ".inV().dedup().count(local)")
        self.assertEqual(execute_plan_local(self.store, _plan(g, out="scalar"),
                                           "A00000001")[0], 2)
        value, _t, err = execute_plan_local(
            self.store, _plan("g.V().hasId({{SEED}}).count('x')", out="scalar"),
            "A00000001")
        self.assertIsNone(value)
        self.assertIn("Invalid count() argument", err)

    def test_malformed_queries_are_isolated(self) -> None:
        scalar_exists = _plan("g.V().hasId({{SEED}}).values('state').sum()", out="scalar",
                              final={"type": "exists", "step": "s1"})
        _value, _trace, error = execute_plan_local(self.store, scalar_exists, "A00000001")
        self.assertIn("must produce id_list", error)
        malformed = _plan("g.V().hasId({{SEED}}).has('state',eq()).id().toList()")
        _value, _trace, error = execute_plan_local(self.store, malformed, "A00000001")
        self.assertIn("requires 1 arguments", error)
        unsupported = _plan("g.V().hasId({{SEED}}).as('x').id().toList()")
        _value, _trace, error = execute_plan_local(self.store, unsupported, "A00000001")
        self.assertIn("outside the closed operator set as()", error)

    def test_public_relation_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_graph(Path(tmp))
            store = GraphStore(root)
            plan = _plan(f"g.V().hasId({{{{SEED}}}}).outE().has('relation','{REL}')"
                         ".inV().dedup().id().toList()")
            self.assertEqual(execute_plan_local(store, plan, "A00000001")[0], 2)
            self.assertEqual(store.relation_meta()[REL]["src"], W)


class EffectivenessTest(unittest.TestCase):
    def test_iv_matches_autobin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_graph(Path(tmp))
            feat = {"feature_name": "f_deg", "feature_name_cn": "Outgoing degree", "window": 0,
                    "logic": "Count accounts reachable through transfers", "expected_value_type": "count",
                    "gremlin_plan": {
                        "steps": [{"id": "s1", "output": "id_list",
                                   "gremlin": f"g.V().hasId({{{{SEED}}}}).outE()"
                                              f".has('relation','{REL}').inV()"
                                              ".dedup().id().toList()"}],
                        "final": {"type": "count", "step": "s1"}}}
            feat_path = root / "features.jsonl"
            feat_path.write_text(json.dumps(feat, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
            rc = eff_main(["--data", str(root), "--features", str(feat_path),
                           "--out-dir", str(root / "out"), "--quiet"])
            self.assertEqual(rc, 0)
            row = json.loads((root / "out" / "effectiveness.jsonl")
                             .read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(row["valid"])
            self.assertEqual(row["coverage"], 1.0)
            df = pd.DataFrame({"value": [2.0, 1.0, 0.0], "label": [1, 0, 0]})
            m, _b = AutobinSplit(smoothing_factor=0.5).split_evaluation(
                df, feature_cols=["value"], label_col="label")
            self.assertAlmostEqual(row["iv"], round(float(m["value"].iv), 6), places=9)

    def test_autobin_standalone(self) -> None:
        df = pd.DataFrame({"value": [0.1, 0.2, 0.2, 0.9, 1.5, 3.0, 3.1, 9.9],
                           "label": [0, 0, 1, 0, 1, 1, 1, 0]})
        metrics, bins = AutobinSplit(smoothing_factor=0.5).split_evaluation(
            df, feature_cols=["value"], label_col="label")
        self.assertGreater(float(metrics["value"].iv), 0.0)
        self.assertTrue(bins["value"])

    def test_iv_smoothing_uses_actual_bin_count(self) -> None:
        counts = [(10, 0), (0, 10), (5, 5)]
        smoothing = 0.5
        total_good = sum(good for good, _bad in counts)
        total_bad = sum(bad for _good, bad in counts)
        denominator_good = total_good + len(counts) * smoothing
        denominator_bad = total_bad + len(counts) * smoothing
        expected = 0.0
        for good, bad in counts:
            p0 = (good + smoothing) / denominator_good
            p1 = (bad + smoothing) / denominator_bad
            expected += (p1 - p0) * float(np.log(p1 / p0))
        actual = AutobinSplit(smoothing_factor=smoothing)._iv_total(counts)
        self.assertAlmostEqual(actual, round(abs(expected), 4), places=9)

    def test_single_class_evaluation_set_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_graph(Path(tmp))
            labels = pd.read_csv(root / "labels.csv")
            labels["label"] = 0
            labels.to_csv(root / "labels.csv", index=False)
            with self.assertRaisesRegex(ValueError, "both label 0 and label 1"):
                load_eval_set(root)

    def test_input_prefix_before_iv_ranking(self) -> None:
        rows = [
            {"feature_name": "first", "valid": True, "iv": 0.1, "coverage": 1.0, "errors": []},
            {"feature_name": "second", "valid": True, "iv": 0.2, "coverage": 1.0, "errors": []},
            {"feature_name": "outside", "valid": True, "iv": 9.0, "coverage": 1.0, "errors": []},
        ]
        summary = effectiveness_summary(rows, pool_limit=2)
        self.assertEqual(summary["features_in"], 3)
        self.assertEqual(summary["pool_size"], 2)
        self.assertEqual(summary["max_iv"], 0.2)
        self.assertAlmostEqual(summary["top10_avg_iv"], 0.15)
        self.assertNotIn("outside", {row["feature_name"] for row in summary["top_features"]})

    def test_ranking_preserves_duplicate_names(self) -> None:
        features = prepare_features([
            {"feature_name": "dup"}, {"feature_name": "dup"}, {"feature_name": "other"}])
        ranking = ranked_indices(features, {"dup#0": 0.1, "dup#1": 0.9, "other#2": 0.8})
        self.assertEqual(ranking, [1, 2, 0])


class DiversityTest(unittest.TestCase):
    def test_identical_signatures(self) -> None:
        schema = {"entities": [{"entity_type": W}],
                  "relations": [{"en": REL, "join_fields": [{"src": W, "dst": W}]}]}
        idx = SchemaIndex(schema)
        plan = {"steps": [{"id": "s1", "output": "id_list",
                           "gremlin": f"g.V().hasId({{{{SEED}}}}).outE()"
                                      f".has('relation','{REL}')"
                                      ".inV().dedup().id().toList()"}],
                "final": {"type": "count", "step": "s1"}}
        a = {"feature_name": "a", "gremlin_plan": plan, "expected_value_type": "count"}
        b = {"feature_name": "b", "gremlin_plan": plan, "expected_value_type": "count"}
        m = batch_metrics([a, b], idx)
        self.assertEqual(m["distinct_structures"], 1)
        self.assertEqual(m["avg_pairwise_sim"], 1.0)
        self.assertEqual(m["redundancy_clusters"], 1)

    def test_isomorphic_graphs_share_canonical_form(self) -> None:
        from fraudfeat.diversity import _TG, _canon, graph_edit_distance

        a = _TG(("Account", "Person", "Device"),
                ((0, 1, "bind", "out"), (0, 2, "use", "out")))
        b = _TG(("Account", "Device", "Person"),
                ((0, 2, "bind", "out"), (0, 1, "use", "out")))
        self.assertEqual(graph_edit_distance(a, b), 0)
        self.assertEqual(_canon(a), _canon(b))

        c = _TG(("Account", "Person", "Person", "Device"),
                ((0, 1, "bind", "out"), (0, 2, "use", "out"),
                 (0, 3, "use", "out"), (1, 2, "know", "out")))
        d = _TG(("Account", "Device", "Person", "Person"),
                ((0, 2, "bind", "out"), (0, 3, "use", "out"),
                 (0, 1, "use", "out"), (2, 3, "know", "out")))
        self.assertEqual(graph_edit_distance(c, d), 0)
        self.assertEqual(_canon(c), _canon(d))

        chain = _TG(("Account", "Person", "Device"),
                    ((0, 1, "bind", "out"), (1, 2, "use", "out")))
        self.assertNotEqual(_canon(a), _canon(chain))

    def test_both_e_and_endpoint_type_constraints(self) -> None:
        schema = {
            "entities": [{"entity_type": W}, {"entity_type": C}],
            "relations": [
                {"en": CUST_REL, "join_fields": [{"src": W, "dst": C}]},
                {"en": REL, "join_fields": [{"src": W, "dst": W}]},
            ],
        }
        idx = SchemaIndex(schema)
        both_plan = _plan(f"g.V().hasId({{{{SEED}}}}).bothE().has('relation','{CUST_REL}')"
                          ".otherV().dedup().id().toList()")
        graph = build_graph({"gremlin_plan": both_plan}, idx)
        self.assertEqual(graph.types, (W, C))
        self.assertEqual(graph.edges, ((0, 1, CUST_REL, "both"),))
        incompatible = {
            "steps": [
                {"id": "s1", "output": "id_list",
                 "gremlin": f"g.V().hasId({{{{SEED}}}}).outE().has('relation','{CUST_REL}')"
                            ".inV().dedup().id().toList()"},
                {"id": "s2", "output": "id_list",
                 "gremlin": f"g.V().hasId({{{{STEP:s1}}}}).outE().has('relation','{REL}')"
                            ".inV().dedup().id().toList()"},
            ],
            "final": {"type": "count", "step": "s2"},
        }
        graph = build_graph({"gremlin_plan": incompatible}, idx)
        self.assertEqual(len(graph.types), 2)
        self.assertEqual(len(graph.edges), 1)

    def test_public_relation_structure(self) -> None:
        schema = {"entities": [{"entity_type": W}],
                  "relations": [{"en": REL, "is_trade": True,
                                 "join_fields": [{"src": W, "dst": W}]}]}
        graph = build_graph({"gremlin_plan": _plan(
            f"g.V().hasId({{{{SEED}}}}).outE().has('relation','{REL}')"
            ".inV().dedup().id().toList()")}, SchemaIndex(schema))
        self.assertEqual(graph.types, (W, W))
        self.assertEqual(graph.edges, ((0, 1, REL, "out"),))


class EmbeddingTest(unittest.TestCase):
    def test_hash_encoder_deterministic(self) -> None:
        vecs = embed_texts(["shared bank card", "shared bank card"], encoder="hash")
        matrix = cosine_matrix(vecs)
        self.assertEqual(matrix.shape, (2, 2))
        self.assertAlmostEqual(float(matrix[0, 1]), 1.0, places=6)
        np.testing.assert_array_equal(np.diag(cosine_matrix([[0.0, 0.0], [1.0, 0.0]])),
                                      np.ones(2))

    def test_remote_response_order_uses_index(self) -> None:
        payload = {"data": [{"index": 1, "embedding": [0.0, 1.0]},
                            {"index": 0, "embedding": [1.0, 0.0]}]}
        with mock.patch("fraudfeat.embedding_diversity._post_json",
                        return_value=(payload, "")):
            vectors = _embed_openai(["first", "second"], base_url="http://unused",
                                    model="test", api_key="", sleep_s=0, retries=0)
        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])


class InterpretabilityTest(unittest.TestCase):
    def test_public_schema_brief(self) -> None:
        schema = {"entities": [{"entity_type": W, "description": "A personal WeChat Pay account.",
                                "fields": [{"name": "state", "dtype": "string",
                                            "values": {"Registered": "Registered",
                                                       "Deregistered": "Deregistered"}}]}],
                  "relations": [{"en": REL, "is_trade": True,
                                 "description": "top_k is the payee's rank among payers.",
                                 "join_fields": [{"src": W, "dst": W}]}]}
        brief = build_domain_brief(schema)
        self.assertIn(REL, brief)
        self.assertIn("marked as fraud", brief)
        self.assertIn("state (string)", brief)
        self.assertIn("Registered=Registered", brief)
        self.assertIn("top_k is the payee's rank among payers.", brief)
        self.assertNotIn("0 edges", brief)

    def test_mock_is_deterministic_and_never_uses_network(self) -> None:
        features = prepare_features([{"feature_name": "a", "logic": "count peers"},
                                     {"feature_name": "b", "logic": "count cards"}])
        requests = build_requests(features, judges=["mock"], brief="")
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("fraudfeat.interpretability.urllib.request.urlopen",
                            side_effect=AssertionError("network access is forbidden")):
                stats = score_requests(requests, Path(tmp), base_url="http://unused",
                                       api_key="", brief="")
            rows = [json.loads(line) for line in
                    (Path(tmp) / "judge_scores.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(stats, {"scored": 2, "skipped": 0, "failed": 0})
        self.assertTrue(all(isinstance(row["score"], int) for row in rows))
        self.assertTrue(all("offline mock" in row["rationale"] for row in rows))

    def test_request_cache_tracks_card_content(self) -> None:
        first = build_requests(prepare_features([{"feature_name": "a", "logic": "count peers"}]),
                               judges=["mock"], brief="")
        changed = build_requests(prepare_features([{"feature_name": "a", "logic": "count cards"}]),
                                 judges=["mock"], brief="")
        self.assertNotEqual(first[0]["request_id"], changed[0]["request_id"])
        with tempfile.TemporaryDirectory() as tmp:
            score_requests(first, Path(tmp), base_url="http://unused", api_key="", brief="")
            stats = score_requests(changed, Path(tmp), base_url="http://unused", api_key="", brief="")
        self.assertEqual(stats["scored"], 1)
        self.assertEqual(stats["skipped"], 0)

    def test_first_scores_fixed_pool_and_duplicate_names(self) -> None:
        features = prepare_features([{"feature_name": "dup"}, {"feature_name": "dup"},
                                     {"feature_name": "dup"}])
        scores = [
            {"entry_key": "dup#0", "feature_name": "dup", "pool_index": 0,
             "judge": "j1", "repeat": 0, "score": 2},
            {"entry_key": "dup#0", "feature_name": "dup", "pool_index": 0,
             "judge": "j1", "repeat": 1, "score": 5},
            {"entry_key": "dup#0", "feature_name": "dup", "pool_index": 0,
             "judge": "j2", "repeat": 0, "score": 4},
            {"entry_key": "dup#1", "feature_name": "dup", "pool_index": 1,
             "judge": "j1", "repeat": 0, "score": 4},
            {"entry_key": "dup#2", "feature_name": "dup", "pool_index": 2,
             "judge": "j1", "repeat": 0, "score": 5},
            {"entry_key": "dup#2", "feature_name": "dup", "pool_index": 2,
             "judge": "j2", "repeat": 0, "score": 5},
        ]
        summary = interpretability_summary(
            features, scores, pool_limit=2,
            iv={"dup#0": 0.1, "dup#1": 0.9, "dup#2": 10.0})
        self.assertEqual(summary["pool_size"], 2)
        self.assertEqual(summary["features_scored"], 2)
        self.assertEqual(summary["main_scores"], 3)
        self.assertEqual(summary["excluded_scores"], 2)
        self.assertEqual(summary["overall_mean"], 3.5)
        self.assertEqual(summary["top10_mean_score"], 3.5)
        self.assertEqual(summary["top10_n"], 2)
        self.assertEqual([row["entry_key"] for row in summary["feature_means"]],
                         ["dup#1", "dup#0"])
        self.assertEqual([row["mean_score"] for row in summary["feature_means"]],
                         [4.0, 3.0])
        self.assertEqual(summary["stability"]["groups"], 1)


class EvaluationPanelTest(unittest.TestCase):
    def test_fixed_panel_loader_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_public_parquet_graph(Path(tmp))
            output = root / "panel.parquet"
            pd.DataFrame({"unique_id": ["A1", "A2"], "label": [1, 0]}).to_parquet(
                output, index=False)
            frame = pd.read_parquet(output)
            self.assertEqual(list(frame.columns), ["unique_id", "label"])
            self.assertEqual(frame["label"].value_counts().to_dict(), {0: 1, 1: 1})
            ids, labels = load_eval_set(root, eval_panel=output)
            self.assertEqual(set(ids), {"A1", "A2"})
            self.assertEqual(labels, dict(zip(frame["unique_id"], frame["label"])))
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                load_eval_set(root, eval_panel=output, limit=1)
            bad = root / "bad_panel.csv"
            pd.DataFrame({"unique_id": ["UNKNOWN", "A2"], "label": [1, 0]}).to_csv(
                bad, index=False)
            with self.assertRaisesRegex(ValueError, "absent from labels"):
                load_eval_set(root, eval_panel=bad)

    def test_disjoint_fixed_panels_and_all_effectiveness_backends(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_public_parquet_graph(Path(tmp))
            offline = root / "offline_panel.parquet"
            pd.DataFrame({"unique_id": ["A1", "A2"], "label": [1, 0]}).to_parquet(
                offline, index=False)
            offline_ids = set(pd.read_parquet(offline)["unique_id"])
            final = root / "evaluation_panel.parquet"
            pd.DataFrame({"unique_id": ["A4", "A3"], "label": [1, 0]}).to_parquet(
                final, index=False)
            final_frame = pd.read_parquet(final)
            self.assertEqual(final_frame["label"].value_counts().to_dict(), {0: 1, 1: 1})
            self.assertTrue(offline_ids.isdisjoint(set(final_frame["unique_id"])))
            examples = Path(__file__).resolve().parents[1] / "examples/features.sql.example.jsonl"
            self.assertEqual(sql_main([
                "--data", str(root), "--features", str(examples),
                "--eval-panel", str(final), "--out-dir", str(root / "sql"), "--quiet"]), 0)
            self.assertEqual(json.loads((root / "sql/effectiveness_summary.json").read_text())
                             ["eval_size"], 2)
            self.assertEqual(eff_main([
                "--data", str(root), "--features", str(root / "features.jsonl"),
                "--eval-panel", str(final), "--out-dir", str(root / "gremlin"),
                "--workers", "1", "--quiet"]), 0)
            self.assertEqual(json.loads((root / "gremlin/effectiveness_summary.json").read_text())
                             ["eval_size"], 2)
            self.assertEqual(direct_main([
                "--data", str(root), "--relations", REL, "--metrics", "degree",
                "--eval-panel", str(final), "--out-dir", str(root / "direct"), "--quiet"]), 0)
            self.assertEqual(json.loads((root / "direct/direct_iv_summary.json").read_text())
                             ["eval_size"], 2)


class SQLEffectivenessTest(unittest.TestCase):
    def test_batch_sql_values_and_gremlin_iv_parity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_public_parquet_graph(Path(tmp))
            eval_ids, labels = load_eval_set(root)
            degree_sql = (
                "SELECT l.unique_id, COUNT(DISTINCT e.dst) AS value "
                "FROM evaluation_accounts AS l LEFT JOIN edges AS e "
                "ON e.src = l.unique_id AND e.relation = '" + REL + "' "
                "GROUP BY l.unique_id")
            feature = {"feature_name": "transfer_out_degree",
                       "expected_value_type": "count", "sql": degree_sql}
            con = connect_release(root, eval_ids, labels, threads=2)
            try:
                values = execute_sql_query(con, degree_sql, eval_ids)
                self.assertEqual(values.tolist(), [2.0, 1.0, 0.0, 1.0])
                sql_row = evaluate_feature_sql(con, feature, eval_ids, labels=labels)
            finally:
                con.close()
            gremlin_feature = prepare_features(
                [json.loads((root / "features.jsonl").read_text().splitlines()[0])])[0]
            store = GraphStore(root, load_labels=False)
            gremlin_row = evaluate_feature(store, gremlin_feature, eval_ids, labels=labels)
            self.assertEqual(sql_row["iv"], gremlin_row["iv"])
            self.assertEqual(sql_row["coverage"], gremlin_row["coverage"])
            self.assertEqual(sql_row["valid"], gremlin_row["valid"])
            self.assertEqual(sql_row["execution_mode"], "batch_panel")

    def test_sql_cli_examples_multi_hop_and_amount(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_public_parquet_graph(Path(tmp))
            examples = Path(__file__).resolve().parents[1] / "examples/features.sql.example.jsonl"
            rc = sql_main(["--data", str(root), "--features", str(examples),
                           "--out-dir", str(root / "sql"), "--threads", "2", "--quiet"])
            self.assertEqual(rc, 0)
            rows = [json.loads(line) for line in
                    (root / "sql/effectiveness.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertTrue(rows[0]["valid"])
            self.assertFalse(rows[1]["valid"])
            self.assertTrue(rows[2]["valid"])
            self.assertTrue(all(row["method"] == "duckdb_sql" for row in rows))
            self.assertTrue(all(not row.get("gremlin_plan") for row in rows))
            summary = json.loads((root / "sql/effectiveness_summary.json").read_text())
            self.assertEqual(summary["backend"], "duckdb_sql")
            self.assertEqual(summary["execution_mode"], "batch_panel")
            self.assertEqual(summary["eval_size"], 4)
            eval_ids, _labels = load_eval_set(root)
            con = connect_release(root, eval_ids, _labels)
            try:
                amount = execute_sql_query(con, json.loads(
                    examples.read_text().splitlines()[2])["sql"], eval_ids)
                self.assertEqual(amount.tolist(), [15.0, 8.0, 0.0, 3.0])
            finally:
                con.close()

    def test_sql_rejects_writes_multiple_statements_and_bad_results(self) -> None:
        with self.assertRaises(ValueError):
            validate_sql("DELETE FROM nodes")
        with self.assertRaises(ValueError):
            validate_sql("SELECT 1 AS value; SELECT 2 AS value")
        with self.assertRaises(ValueError):
            validate_sql("SELECT * FROM read_parquet('other.parquet')")
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_public_parquet_graph(Path(tmp))
            eval_ids, labels = load_eval_set(root)
            con = connect_release(root, eval_ids, labels)
            try:
                feature = {"feature_name": "incomplete", "sql":
                           "SELECT 'A1' AS unique_id, 1 AS value"}
                row = evaluate_feature_sql(con, feature, eval_ids, labels=labels)
                self.assertFalse(row["valid"])
                self.assertIn("missing 3", row["errors"][0])
                wrong_columns = {"feature_name": "wrong_columns", "sql":
                                 "SELECT unique_id FROM evaluation_accounts"}
                row = evaluate_feature_sql(con, wrong_columns, eval_ids, labels=labels)
                self.assertIn("unique_id, value", row["errors"][0])
            finally:
                con.close()

    def test_sql_labels_are_not_registered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_public_parquet_graph(Path(tmp))
            eval_ids, labels = load_eval_set(root)
            con = connect_release(root, eval_ids, labels)
            try:
                cursor = con.execute("SELECT * FROM evaluation_accounts LIMIT 0")
                self.assertEqual([item[0] for item in cursor.description], ["unique_id"])
                tables = {str(row[0]) for row in con.execute(
                    "SELECT table_name FROM duckdb_tables()").fetchall()}
                self.assertFalse(any("label" in name for name in tables))
                with self.assertRaises(Exception):
                    con.execute("SELECT * FROM labels")
                leaked = {"feature_name": "label_leak", "sql":
                          "SELECT unique_id, CAST(label AS DOUBLE) AS value "
                          "FROM evaluation_accounts"}
                row = evaluate_feature_sql(con, leaked, eval_ids, labels=labels)
                self.assertFalse(row["valid"])
                self.assertFalse(row["exec_ok"])
            finally:
                con.close()

    def test_sql_query_is_in_embedding_text(self) -> None:
        raw = json.loads((Path(__file__).resolve().parents[1] /
                          "examples/features.sql.example.jsonl").read_text().splitlines()[0])
        text = feature_text(prepare_features([raw])[0])
        self.assertIn("Query:", text)
        self.assertIn("SELECT", text)
        self.assertIn(REL, text)
        self.assertIn(raw["sql"], text)

    def test_graph_pattern_matches_gremlin_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_public_parquet_graph(Path(tmp))
            schema = json.loads((root / "schema.json").read_text())
            idx = SchemaIndex(schema)
            gremlin_rows = [json.loads(line) for line in
                            (root / "features.jsonl").read_text().splitlines()]
            sql_rows = [json.loads(line) for line in
                        (Path(__file__).resolve().parents[1] /
                         "examples/features.sql.example.jsonl").read_text().splitlines()[:2]]
            for gremlin, sql in zip(gremlin_rows, sql_rows):
                self.assertEqual(build_graph(gremlin, idx), build_graph(sql, idx))
                self.assertEqual(stat_signature(gremlin, idx), stat_signature(sql, idx))

    def test_csv_release_and_raw_amount_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_graph(Path(tmp))
            eval_ids, labels = load_eval_set(root)
            query = (
                "SELECT l.unique_id, COALESCE(SUM(e.total_amount_yuan), 0.0) AS value "
                "FROM evaluation_accounts AS l LEFT JOIN edges AS e "
                "ON e.src = l.unique_id AND e.relation = '" + REL + "' "
                "GROUP BY l.unique_id")
            con = connect_release(root, eval_ids, labels)
            try:
                values = execute_sql_query(con, query, eval_ids)
            finally:
                con.close()
            self.assertEqual(values.tolist(), [140.0, 200.0, 0.0])


class ReleaseIntegrationTest(unittest.TestCase):
    def test_parquet_release_effectiveness_and_public_enums(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_public_parquet_graph(Path(tmp))
            store = GraphStore(root)
            self.assertEqual(store.node_value("A1", "entity_type"), W)
            self.assertEqual(store.node_value("N2", "entity_type"), "natural_person")
            self.assertEqual(store.node_value("N2", "gender"), "Female")
            registered = _plan("g.V().hasId({{SEED}})"
                               f".has('entity_type','{W}')"
                               ".has('state','Registered').dedup().id().toList()")
            value, _trace, error = execute_plan_local(store, registered, "A1")
            self.assertEqual(value, 1, error)
            self.assertEqual(execute_plan_local(store, registered, "A3")[0], 0)
            rc = eff_main(["--data", str(root), "--features", str(root / "features.jsonl"),
                           "--out-dir", str(root / "effectiveness"), "--quiet"])
            self.assertEqual(rc, 0)
            rows = [json.loads(line) for line in
                    (root / "effectiveness/effectiveness.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[0]["valid"])
            self.assertEqual(rows[0]["coverage"], 1.0)
            self.assertGreater(rows[0]["iv"], 0.0)
            self.assertFalse(rows[1]["valid"], rows[1])
            self.assertTrue(any("constant" in error for error in rows[1]["errors"]), rows[1])
            schema = json.loads((root / "schema.json").read_text())
            structural = batch_metrics(
                [json.loads(line) for line in (root / "features.jsonl").read_text().splitlines()],
                SchemaIndex(schema))
            self.assertEqual(structural["n"], 2)
            self.assertEqual(structural["distinct_structures"], 2)

    def test_direct_iv_without_gremlin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_public_parquet_graph(Path(tmp))
            rc = direct_main([
                "--data", str(root), "--relations", REL,
                "--metrics", "degree,edge_count,amount_sum",
                "--out-dir", str(root / "direct"), "--quiet"])
            self.assertEqual(rc, 0)
            rows = [json.loads(line) for line in
                    (root / "direct/direct_iv.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(row["method"] == "direct_table_aggregation" for row in rows))
            self.assertTrue(all("gremlin_plan" not in row for row in rows))
            degree = next(row for row in rows
                          if row["role"] == "src" and row["metric"] == "degree")
            frame = pd.DataFrame({"value": [2.0, 1.0, 0.0, 1.0], "label": [1, 0, 0, 1]})
            metrics, _bins = AutobinSplit(smoothing_factor=0.5).split_evaluation(
                frame, feature_cols=["value"], label_col="label")
            self.assertAlmostEqual(degree["iv"], float(metrics["value"].iv), places=6)
            summary = json.loads((root / "direct/direct_iv_summary.json").read_text())
            self.assertEqual(summary["method"], "direct_table_aggregation")

    def test_mock_and_hash_cli_on_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_public_parquet_graph(Path(tmp))
            self.assertEqual(eff_main([
                "--data", str(root), "--features", str(root / "features.jsonl"),
                "--out-dir", str(root / "effectiveness"), "--quiet"]), 0)
            brief_path = root / "nested/domain/brief.txt"
            self.assertEqual(interpret_main([
                "--features", str(root / "features.jsonl"), "--schema", str(root / "schema.json"),
                "--judges", "mock", "--out-dir", str(root / "judge"),
                "--save-brief", str(brief_path), "--quiet"]), 0)
            brief = brief_path.read_text(encoding="utf-8")
            self.assertIn("Registered=Registered", brief)
            self.assertIn("top_k is the payee's rank", brief)
            self.assertNotIn("0 edges", brief)
            judge_summary = json.loads((root / "judge/interpretability_summary.json").read_text())
            self.assertEqual(judge_summary["features_scored"], 2)
            embedding_args = [
                "--features", str(root / "features.jsonl"), "--encoder", "hash",
                "--out-dir", str(root / "embedding"),
                "--iv-jsonl", str(root / "effectiveness/effectiveness.jsonl"), "--quiet"]
            self.assertEqual(embedding_main(embedding_args), 0)
            output_path = root / "embedding/embedding_diversity.json"
            first = json.loads(output_path.read_text())
            self.assertEqual(first["n_features"], 2)
            self.assertTrue(first["texts_sha1"] and first["iv_sha1"])
            feature_rows = [json.loads(line) for line in
                            (root / "features.jsonl").read_text().splitlines()]
            feature_rows[0]["explanation"] = "Changed explanation requiring a new embedding."
            (root / "features.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in feature_rows), encoding="utf-8")
            self.assertEqual(embedding_main(embedding_args), 0)
            changed_text = json.loads(output_path.read_text())
            self.assertNotEqual(first["texts_sha1"], changed_text["texts_sha1"])
            iv_path = root / "effectiveness/effectiveness.jsonl"
            iv_rows = [json.loads(line) for line in iv_path.read_text().splitlines()]
            iv_rows[0]["iv"] = 0.123456
            iv_path.write_text("".join(json.dumps(row) + "\n" for row in iv_rows), encoding="utf-8")
            self.assertEqual(embedding_main(embedding_args), 0)
            changed_iv = json.loads(output_path.read_text())
            self.assertNotEqual(changed_text["iv_sha1"], changed_iv["iv_sha1"])


class ReleaseContractTest(unittest.TestCase):
    def test_schema_identifiers_match_the_authoritative_release_contract(self) -> None:
        default_root = Path(__file__).resolve().parents[2] / "data"
        path = Path(os.environ.get("FRAUDFEAT_DATA", str(default_root))) / "schema.json"
        if not path.is_file():
            self.skipTest("Download the data release to run this schema contract test")
        schema = json.loads(path.read_text(encoding="utf-8"))
        expected_entities = {
            W: ["authen_time", "mobile_update_time", "reg_time", "state", "dereg_time"],
            "merchant_pay_account": [
                "apply_attr", "register_channel", "register_status", "register_time",
                "subject_type", "business_scene"],
            C: ["cre_type", "first_authen_time", "first_register_time", "last_dereg_time"],
            "natural_person": ["birthday", "cre_end_date", "cre_start_date", "cre_type",
                               "gender"],
            "bank_card": [],
            "phone_number": [],
            "wechat_pay_bill_deletion_event": ["operation_type"],
            "wechat_pay_card_back_information_query_event": ["bank_card_type"],
        }
        actual_entities = {
            entity["entity_type"]: [field["name"] for field in entity.get("fields") or []]
            for entity in schema["entities"]
        }
        self.assertEqual(actual_entities, expected_entities)
        self.assertTrue(all("vtype" not in entity and "en" not in entity
                            for entity in schema["entities"]))
        expected_relations = [
            ("personal_wechat_pay_transfer_payee_side", W, W),
            ("personal_wechat_pay_transfer_payer_side", W, W),
            ("merchant_payment_receipt_payer_side", W, "merchant_pay_account"),
            ("merchant_payment_receipt_payee_side", "merchant_pay_account", W),
            ("wechat_pay_withdrawal_to_bank_card", W, "bank_card"),
            ("wechat_pay_account_top_up_with_bank_card", W, "bank_card"),
            ("wechat_pay_red_packet_funded_by_bank_card", W, "bank_card"),
            ("merchant_refund_payee_side", W, "merchant_pay_account"),
            ("merchant_refund_payer_side", "merchant_pay_account", W),
            ("wechat_pay_account_transfer_to_bank_card", W, "bank_card"),
            ("wechat_pay_account_bind_individual_customer", W, C),
            ("wechat_pay_account_bind_natural_person", W, "natural_person"),
            ("wechat_pay_bill_deletion", W, "wechat_pay_bill_deletion_event"),
            ("wechat_pay_account_bind_phone_number", W, "phone_number"),
            ("natural_person_bind_individual_customer", "natural_person", C),
            ("personal_wechat_pay_account_is_merchant_super_administrator", W,
             "merchant_pay_account"),
            ("wechat_pay_account_bind_bank_card", W, "bank_card"),
            ("wechat_pay_card_back_information_query", W,
             "wechat_pay_card_back_information_query_event"),
        ]
        actual_relations = [
            (relation["en"], relation["join_fields"][0]["src"], relation["join_fields"][0]["dst"])
            for relation in sorted(schema["relations"], key=lambda item: item["index"])
        ]
        self.assertEqual(actual_relations, expected_relations)
        self.assertTrue(all("jf" not in endpoint and "source_join_fields" not in relation
                            for relation in schema["relations"]
                            for endpoint in relation["join_fields"]))


class ReleaseTextTest(unittest.TestCase):
    def test_python_sources_contain_no_chinese(self) -> None:
        root = Path(__file__).resolve().parents[1]
        violations = [str(path.relative_to(root)) for path in root.rglob("*.py")
                      if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]",
                                   path.read_text(encoding="utf-8"))]
        self.assertEqual(violations, [])

    def test_public_prompt_docs_and_examples_are_english(self) -> None:
        root = Path(__file__).resolve().parents[1]
        paths = [root / "README.md", root / "examples" / "features.example.jsonl",
                 root / "examples" / "features.sql.example.jsonl",
                 root / "fraudfeat" / "interpretability.py"]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        self.assertIsNone(re.search(r"[\u4e00-\u9fff]", combined))
        self.assertIsNone(re.search(
            r"(?i)(?:api[_-]?key|token|secret)\s*=\s*['\"][^'\"]+['\"]", combined))

    def test_examples_have_consistent_transfer_and_ratio_semantics(self) -> None:
        path = Path(__file__).resolve().parents[1] / "examples" / "features.example.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        ratio = next(row for row in rows if row["expected_value_type"] == "ratio")
        self.assertIn("counterparty", ratio["feature_name"])
        self.assertIn("payee accounts", ratio["logic"])
        for row in rows:
            if "transfer" in row["feature_name"]:
                gremlin = " ".join(step["gremlin"] for step in row["gremlin_plan"]["steps"])
                self.assertIn("'relation','personal_wechat_pay_transfer_payer_side'", gremlin)
                self.assertNotIn("join_field", gremlin)


if __name__ == "__main__":
    unittest.main(verbosity=2)
