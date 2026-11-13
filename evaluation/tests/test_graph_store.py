"""Regression tests for the read-only graph indexes and traversal accessors."""

from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from fraudfeat.graph_store import GraphStore, _AdjacencyIndex


class GraphStoreIndexTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        schema = {
            "format": "csv",
            "entities": [{"entity_type": "account", "file": "accounts.csv"}],
            "relations": [{"en": "transfer", "file": "transfers.csv"}],
            "labels": {"file": "labels.csv"},
        }
        (self.root / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
        pd.DataFrame({
            "unique_id": ["a", "b", "c", "isolated"],
            "entity_type": ["account"] * 4,
            "state": ["Registered", "Deregistered", "", "Registered"],
            "code": ["001", "0", "00000000", ""],
        }).to_csv(self.root / "accounts.csv", index=False)
        pd.DataFrame({
            "src": ["a", "c", "a", "a", "a", "b"],
            "dst": ["b", "a", "a", "b", "c", "a"],
            "relation": ["transfer"] * 6,
            "total_amount": [100, 200, 0, 350, None, 1],
            "trade_count": [1, 2, 1, 4, 3, 1],
            "top_k": [1, 1, 2, 3, None, 1],
        }).to_csv(self.root / "transfers.csv", index=False)
        pd.DataFrame({"unique_id": ["a", "b"], "label": [1, 0]}).to_csv(
            self.root / "labels.csv", index=False)
        self.store = GraphStore(self.root, load_labels=True)

    def assertValuesEqual(self, actual, expected):
        pd.testing.assert_series_equal(
            pd.Series(actual), pd.Series(expected), check_dtype=False,
            check_names=False)

    def test_out_in_and_self_loop_order(self):
        np.testing.assert_array_equal(self.store.incident("a", "out"), [0, 2, 3, 4])
        np.testing.assert_array_equal(self.store.incident("a", "in"), [1, 2, 5])
        np.testing.assert_array_equal(
            self.store.incident("a"), [0, 2, 3, 4, 1, 2, 5])

    def test_unknown_and_isolated_vertices(self):
        for uid in ("unknown", "isolated"):
            for direction in ("in", "out", "both"):
                result = self.store.incident(uid, direction)
                self.assertEqual(result.dtype, np.dtype("int64"))
                self.assertEqual(result.size, 0)
        self.assertFalse(self.store.has_vertex("unknown"))
        self.assertTrue(self.store.has_vertex("isolated"))

    def test_endpoint_order_duplicates_and_negative_positions(self):
        positions = np.array([4, 0, 4, -1], dtype=np.int64)
        for side in ("src", "other_src", "dst", "other_dst"):
            column = "src" if side in ("src", "other_src") else "dst"
            expected = self.store.edges.iloc[positions][column].to_numpy(dtype=object)
            self.assertValuesEqual(self.store.edge_endpoint(positions, side), expected)

    def test_property_values_match_dataframe_selection(self):
        positions = np.array([4, 2, 0, 5, 0], dtype=np.int64)
        for prop in ("relation", "src", "dst", "total_amount", "trade_count", "top_k", "edge_id"):
            expected = self.store.edges.iloc[positions][prop].to_numpy()
            self.assertValuesEqual(self.store.edge_values(positions, prop), expected)

    def test_generated_edge_ids_and_alias(self):
        positions = np.array([5, 0, 2], dtype=np.int64)
        expected = ["00000000000000000006", "00000000000000000001", "00000000000000000003"]
        for prop in ("id", "edge_id"):
            self.assertValuesEqual(self.store.edge_values(positions, prop), expected)

    def test_missing_properties_and_empty_selections(self):
        self.assertValuesEqual(self.store.edge_values(np.array([1, 0]), "absent"), [None, None])
        for prop in ("relation", "trade_count", "absent"):
            result = self.store.edge_values(np.array([], dtype=np.int64), prop)
            self.assertEqual(result.dtype, np.dtype(object))
            self.assertEqual(result.size, 0)
        self.assertEqual(self.store.edge_endpoint(np.array([], dtype=np.int64), "dst").size, 0)

    def test_hot_accessors_do_not_slice_dataframe_rows(self):
        positions = np.array([0, 2], dtype=np.int64)
        def forbidden(_frame):
            raise AssertionError("Traversal must not materialize DataFrame rows")
        with patch.object(pd.DataFrame, "iloc", property(forbidden)):
            self.assertValuesEqual(self.store.edge_endpoint(positions, "dst"), ["b", "a"])
            self.assertValuesEqual(self.store.edge_values(positions, "relation"), ["transfer", "transfer"])
            self.assertValuesEqual(self.store.edge_values(positions, "trade_count"), [1, 1])
        cached = self.store._edge_prop_cache["relation"]
        self.store.edge_values(positions, "relation")
        self.assertIs(cached, self.store._edge_prop_cache["relation"])

    def test_node_scalars_preserve_strings_and_type_fallback(self):
        self.assertEqual(self.store.node_value("a", "code"), "001")
        self.assertEqual(self.store.node_value("b", "code"), "0")
        self.assertEqual(self.store.node_value("c", "state"), "")
        for prop in ("label", "entity_type"):
            self.assertEqual(self.store.node_value("a", prop), "account")
        for prop in ("id", "unique_id"):
            self.assertEqual(self.store.node_value("a", prop), "a")
        self.assertIsNone(self.store.node_value("unknown", "id"))
        self.assertIsNone(self.store.node_value("a", "absent"))
        self.store.nodes["account"].at["a", "entity_type"] = None
        self.assertEqual(self.store.node_value("a", "entity_type"), "account")

    def test_edge_rows_labels_and_summary_unchanged(self):
        pd.testing.assert_frame_equal(self.store.edge_rows(np.array([3, 0])), self.store.edges.iloc[[3, 0]])
        self.assertIs(self.store.edge_rows(), self.store.edges)
        self.assertEqual(self.store.edge_rows(np.array([], dtype=int)).shape[0], 0)
        self.assertEqual(self.store.labels, {"a": 1, "b": 0})
        self.assertEqual(self.store.summary()["nodes"], 4)
        self.assertEqual(self.store.summary()["edges"], 6)

    def test_lazy_and_no_edge_loading(self):
        lazy = GraphStore(self.root, build_adjacency=False)
        self.assertEqual(lazy.incident("a").size, 0)
        lazy._build_adjacency()
        np.testing.assert_array_equal(lazy.incident("a"), self.store.incident("a"))
        no_edges = GraphStore(self.root, load_edges=False)
        self.assertEqual(no_edges.edges.shape[0], 0)
        self.assertEqual(no_edges.incident("a").size, 0)

    def test_pickle_roundtrip(self):
        self.store.edge_values(np.array([0]), "trade_count")
        restored = pickle.loads(pickle.dumps(self.store))
        np.testing.assert_array_equal(restored.incident("a"), self.store.incident("a"))
        self.assertValuesEqual(restored.edge_values(np.array([4, 0]), "top_k"), [np.nan, 1])
        self.assertEqual(restored.node_value("a", "code"), "001")

    def test_rebuild_invalidates_column_cache(self):
        self.store.edge_values(np.array([0]), "trade_count")
        self.store.edges = self.store.edges.copy()
        self.store.edges.loc[0, "trade_count"] = 99
        self.store.edges.index = pd.Index([10, 20, 30, 40, 50, 60])
        self.store._build_adjacency()
        self.assertValuesEqual(self.store.edge_values(np.array([0]), "trade_count"), [99])
        self.assertEqual(list(self.store.edges.index), list(range(6)))

    def test_extension_array_columns(self):
        self.store.edges["relation"] = self.store.edges["relation"].astype("string")
        self.store.edges["trade_count"] = self.store.edges["trade_count"].astype("Int64")
        self.store.edges.loc[4, "trade_count"] = pd.NA
        self.store._build_adjacency()
        for positions in (np.array([4, 0]), np.array([1, 2])):
            for prop in ("relation", "trade_count"):
                self.assertValuesEqual(
                    self.store.edge_values(positions, prop),
                    self.store.edges.iloc[positions][prop].to_numpy())

    def test_nullable_large_integers_do_not_lose_precision(self):
        large = 2**60 + 3
        self.store.edges["trade_count"] = pd.array(
            [large, 2, 3, 4, None, 6], dtype="Int64")
        result = self.store.edge_values(np.array([0]), "trade_count")
        self.assertEqual(int(result[0]), large)
        for positions in (np.array([0]), np.array([0, 4]), np.array([-1, 0])):
            self.assertValuesEqual(
                self.store.edge_values(positions, "trade_count"),
                self.store.edges.iloc[positions]["trade_count"].to_numpy())

    def test_index_matches_groupby_for_random_endpoints(self):
        rng = np.random.default_rng(42)
        for dtype in (object, "string", "category"):
            for size in (0, 1, 31, 1024):
                with self.subTest(dtype=dtype, size=size):
                    series = pd.Series(rng.choice(["c", "a", "b", None], size=size), dtype=dtype)
                    frame = pd.DataFrame({"endpoint": series})
                    expected = frame.groupby("endpoint", sort=False, observed=True).indices
                    index = _AdjacencyIndex(series)
                    self.assertEqual(set(index), {str(k) for k in expected})
                    for key, rows in expected.items():
                        np.testing.assert_array_equal(index[str(key)], rows)
                    self.assertIsNone(index.get("missing"))

    def test_index_string_key_collision_matches_old_mapping(self):
        series = pd.Series([1, "1", 2, "1", 1], dtype=object)
        groups = pd.DataFrame({"endpoint": series}).groupby("endpoint", sort=False).indices
        expected = {str(key): rows for key, rows in groups.items()}
        index = _AdjacencyIndex(series)
        for key, rows in expected.items():
            np.testing.assert_array_equal(index[key], rows)


if __name__ == "__main__":
    unittest.main()
