from __future__ import annotations

import gc
import importlib.metadata
import inspect
import unittest

import egraph as eg
import numpy as np


class TestCsrGraph(unittest.TestCase):
    def test_from_csr_undirected_and_directed_sanitization(self) -> None:
        # Undirected CSR with self-loop (0, 0), duplicate/asymmetric weights, and missing reverse edge
        indptr = np.array([0, 3, 4, 4], dtype=np.uint32)
        indices = np.array([0, 1, 1, 2], dtype=np.uint32)
        weights = np.array([9.0, 1.5, 3.5, 2.0], dtype=np.float64)

        g = eg.CsrGraph.from_csr(
            indptr, indices, weights=weights, directed=False, sanitize=True
        )
        self.assertEqual(g.num_nodes, 3)
        self.assertEqual(g.num_edges, 2)
        self.assertFalse(g.is_directed)
        np.testing.assert_array_equal(g.indptr(), np.array([0, 1, 3, 4], dtype=np.uint32))
        np.testing.assert_array_equal(g.indices(), np.array([1, 0, 2, 1], dtype=np.uint32))
        w = g.weights()
        assert w is not None
        np.testing.assert_allclose(w, np.array([3.5, 3.5, 2.0, 2.0], dtype=np.float64))

        # Directed CSR from Python lists with sanitization
        gd = eg.CsrGraph.from_csr(
            [0, 3, 3],
            [0, 1, 1],
            weights=[5.0, 1.0, 4.0],
            directed=True,
            sanitize=True,
        )
        self.assertEqual(gd.num_nodes, 2)
        self.assertEqual(gd.num_edges, 1)
        self.assertTrue(gd.is_directed)
        np.testing.assert_array_equal(gd.indptr(), np.array([0, 1, 1], dtype=np.uint32))
        np.testing.assert_array_equal(gd.indices(), np.array([1], dtype=np.uint32))
        wd = gd.weights()
        assert wd is not None
        np.testing.assert_allclose(wd, np.array([4.0], dtype=np.float64))

    def test_from_edges_unweighted_weighted_and_non_contiguous(self) -> None:
        raw = np.array([[0, 1], [99, 99], [1, 2], [88, 88], [0, 0]], dtype=np.int64)
        sliced = raw[::2, :]
        self.assertFalse(sliced.flags["C_CONTIGUOUS"])

        g = eg.CsrGraph.from_edges(3, sliced, directed=False, sanitize=True)
        self.assertEqual(g.num_nodes, 3)
        self.assertEqual(g.num_edges, 2)
        self.assertIsNone(g.weights())
        np.testing.assert_array_equal(g.indptr(), np.array([0, 1, 3, 4], dtype=np.uint32))
        np.testing.assert_array_equal(g.indices(), np.array([1, 0, 2, 1], dtype=np.uint32))

        # Weighted from Python tuple sequence + float32 weights
        gw = eg.CsrGraph.from_edges(
            3,
            [(0, 1), (1, 2)],
            weights=np.array([1.25, 2.5], dtype=np.float32),
            directed=True,
            sanitize=False,
        )
        self.assertEqual(gw.num_nodes, 3)
        self.assertEqual(gw.num_edges, 2)
        self.assertTrue(gw.is_directed)
        ww = gw.weights()
        assert ww is not None
        np.testing.assert_allclose(ww, np.array([1.25, 2.5], dtype=np.float64))

        # Empty edge list
        g_empty = eg.CsrGraph.from_edges(4, np.empty((0, 2), dtype=np.uint32))
        self.assertEqual(g_empty.num_nodes, 4)
        self.assertEqual(g_empty.num_edges, 0)
        self.assertIsNone(g_empty.weights())

    def test_largest_connected_component(self) -> None:
        # Component {0, 1} of size 2, component {2, 3, 4, 5} of size 4, isolated node 6
        g = eg.CsrGraph.from_edges(
            7,
            [(0, 1), (2, 3), (3, 4), (4, 5)],
            weights=[1.0, 2.0, 3.0, 4.0],
            directed=False,
            sanitize=True,
        )
        lcc, orig_ids = g.largest_connected_component()
        self.assertEqual(lcc.num_nodes, 4)
        self.assertEqual(lcc.num_edges, 3)
        self.assertEqual(orig_ids.dtype, np.uint32)
        np.testing.assert_array_equal(orig_ids, np.array([2, 3, 4, 5], dtype=np.uint32))
        np.testing.assert_array_equal(lcc.indptr(), np.array([0, 1, 3, 5, 6], dtype=np.uint32))
        np.testing.assert_array_equal(lcc.indices(), np.array([1, 0, 2, 1, 3, 2], dtype=np.uint32))
        w_lcc = lcc.weights()
        assert w_lcc is not None
        np.testing.assert_allclose(
            w_lcc, np.array([2.0, 2.0, 3.0, 3.0, 4.0, 4.0], dtype=np.float64)
        )

        # Weakly connected component on directed graph
        gd = eg.CsrGraph.from_edges(5, [(0, 1), (2, 3), (4, 3)], directed=True, sanitize=True)
        lcc_d, orig_d = gd.largest_connected_component()
        self.assertEqual(lcc_d.num_nodes, 3)
        self.assertEqual(lcc_d.num_edges, 2)
        self.assertTrue(lcc_d.is_directed)
        np.testing.assert_array_equal(orig_d, np.array([2, 3, 4], dtype=np.uint32))

        # Empty graph LCC
        g0 = eg.CsrGraph.from_edges(0, [])
        lcc0, orig0 = g0.largest_connected_component()
        self.assertEqual(lcc0.num_nodes, 0)
        self.assertEqual(len(orig0), 0)

    def test_csr_graph_adversarial_errors(self) -> None:
        # Mismatched lengths
        with self.assertRaises(ValueError):
            eg.CsrGraph.from_csr([0, 1, 2], [1, 0], weights=[1.0])
        with self.assertRaises(ValueError):
            eg.CsrGraph.from_csr([0, 1, 1], [1], weights=[1.0, 2.0])
        with self.assertRaises(ValueError):
            eg.CsrGraph.from_edges(3, [(0, 1), (1, 2)], weights=[1.0])
        with self.assertRaises(ValueError):
            eg.CsrGraph.from_edges(3, np.zeros((2, 3), dtype=np.int64))
        with self.assertRaises(ValueError):
            eg.CsrGraph.from_edges(3, np.zeros((4,), dtype=np.int64))
        with self.assertRaises(ValueError):
            eg.CsrGraph.from_csr([], [])
        with self.assertRaises(ValueError):
            eg.CsrGraph.from_csr([1, 1], [0])
        with self.assertRaises(ValueError):
            eg.CsrGraph.from_csr([0, 2, 1], [0])
        with self.assertRaises(ValueError):
            eg.CsrGraph.from_csr([0, 1, 2], [1, 0], sanitize=False, weights=[1.0, 2.0])

        # OOB indices -> IndexError
        with self.assertRaises(IndexError):
            eg.CsrGraph.from_csr([0, 1, 1], [5])
        with self.assertRaises(IndexError):
            eg.CsrGraph.from_edges(3, [(0, 3)])

        # Negative / overflowing indices -> ValueError
        with self.assertRaises(ValueError):
            eg.CsrGraph.from_edges(3, np.array([[-1, 1]], dtype=np.int64))
        with self.assertRaises(ValueError):
            eg.CsrGraph.from_csr([0, -1], [])

        # NaN / Inf weights -> ValueError
        for bad_w in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                eg.CsrGraph.from_csr([0, 1, 2], [1, 0], weights=[bad_w, 1.0])
            with self.assertRaises(ValueError):
                eg.CsrGraph.from_edges(2, [(0, 1)], weights=[bad_w])

        # Wrong types -> TypeError (including bool and float indices)
        with self.assertRaises(TypeError):
            eg.CsrGraph.from_csr(np.array([0.0, 1.0]), [0])
        with self.assertRaises(TypeError):
            eg.CsrGraph.from_csr([0, 1], [True])
        with self.assertRaises(TypeError):
            eg.CsrGraph.from_edges(3, np.array([[0.0, 1.0]], dtype=np.float64))
        with self.assertRaises(TypeError):
            eg.CsrGraph.from_edges(2, [(0, 1)], weights=[True])
        with self.assertRaises(TypeError):
            eg.CsrGraph.from_edges(2, [(0, 1)], weights="bad")  # type: ignore[arg-type]


class TestLayout2d(unittest.TestCase):
    def test_zero_copy_roundtrip_f64_and_f32(self) -> None:
        for dtype_arg, np_dtype, expected_str in (
            ("float64", np.float64, "float64"),
            ("f64", np.float64, "float64"),
            ("float32", np.float32, "float32"),
            ("f32", np.float32, "float32"),
            (np.float32, np.float32, "float32"),
            (np.dtype("float64"), np.float64, "float64"),
        ):
            layout = eg.Layout2d.zeros(4, dtype=dtype_arg)
            self.assertEqual(len(layout), 4)
            self.assertEqual(layout.dtype, expected_str)
            self.assertTrue(layout.is_all_finite())

            view = layout.coords(copy=False)
            self.assertEqual(view.shape, (4, 2))
            self.assertEqual(view.dtype, np_dtype)
            self.assertTrue(view.flags["WRITEABLE"])
            self.assertFalse(view.flags["OWNDATA"])
            self.assertIs(view.base, layout)

            # Mutate via NumPy zero-copy view and verify layout observes the update
            view[0, 0] = 1.5
            view[2, 1] = -3.25
            view2 = layout.coords(copy=False)
            self.assertTrue(np.shares_memory(view, view2))
            self.assertEqual(float(view2[0, 0]), 1.5)
            self.assertEqual(float(view2[2, 1]), -3.25)

            # copy=True produces an isolated copy
            copied = layout.coords(copy=True)
            self.assertFalse(np.shares_memory(view, copied))
            self.assertEqual(copied.dtype, np_dtype)
            copied[0, 0] = 999.0
            self.assertEqual(float(layout.coords(copy=False)[0, 0]), 1.5)

            # Writing NaN through the view reflects in is_all_finite()
            view[3, 1] = float("nan")
            self.assertFalse(layout.is_all_finite())

    def test_view_keeps_backing_buffer_alive_after_layout_deleted(self) -> None:
        for dtype_str, np_dtype in (("float64", np.float64), ("float32", np.float32)):
            layout = eg.Layout2d.from_coords(
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np_dtype)
            )
            self.assertEqual(layout.dtype, dtype_str)
            view = layout.coords(copy=False)
            del layout
            gc.collect()
            view[1, 1] = 8.5
            np.testing.assert_allclose(
                view, np.array([[1.0, 2.0], [3.0, 8.5]], dtype=np_dtype)
            )

    def test_from_coords_and_empty_layouts(self) -> None:
        # Fortran-contiguous float32 input
        arr_f = np.asfortranarray(
            np.array([[0.5, -1.5], [2.25, 3.75], [-4.0, 0.0]], dtype=np.float32)
        )
        l32 = eg.Layout2d.from_coords(arr_f)
        self.assertEqual(len(l32), 3)
        self.assertEqual(l32.dtype, "float32")
        np.testing.assert_allclose(l32.coords(), arr_f)

        # Python list input -> float64
        l64 = eg.Layout2d.from_coords([[1, 2], [3.5, -4.5]])
        self.assertEqual(len(l64), 2)
        self.assertEqual(l64.dtype, "float64")
        np.testing.assert_allclose(
            l64.coords(), np.array([[1.0, 2.0], [3.5, -4.5]], dtype=np.float64)
        )

        # Empty (0, 2) layouts for both dtypes
        for dtype_str, np_dtype in (("float64", np.float64), ("float32", np.float32)):
            empty = eg.Layout2d.zeros(0, dtype=dtype_str)
            self.assertEqual(len(empty), 0)
            self.assertTrue(empty.is_all_finite())
            v = empty.coords(copy=False)
            self.assertEqual(v.shape, (0, 2))
            self.assertEqual(v.dtype, np_dtype)

    def test_layout2d_adversarial_errors(self) -> None:
        # Invalid dtype strings / types
        for bad_dtype in ("int32", "float16", "invalid", np.int64):
            with self.assertRaises(ValueError):
                eg.Layout2d.zeros(4, dtype=bad_dtype)  # type: ignore[arg-type]
        for bad_dtype_type in (None, 123, True):
            with self.assertRaises(TypeError):
                eg.Layout2d.zeros(4, dtype=bad_dtype_type)  # type: ignore[arg-type]

        # Negative / invalid n
        with self.assertRaises(ValueError):
            eg.Layout2d.zeros(-1)
        with self.assertRaises(TypeError):
            eg.Layout2d.zeros(True)
        with self.assertRaises(TypeError):
            eg.Layout2d.zeros(1.5)  # type: ignore[arg-type]

        # Bad shapes for from_coords
        with self.assertRaises(ValueError):
            eg.Layout2d.from_coords(np.zeros((4,), dtype=np.float64))
        with self.assertRaises(ValueError):
            eg.Layout2d.from_coords(np.zeros((4, 3), dtype=np.float64))
        with self.assertRaises(ValueError):
            eg.Layout2d.from_coords([[1.0, 2.0, 3.0]])

        # NaN / Inf coords
        for bad_val in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                eg.Layout2d.from_coords(np.array([[0.0, bad_val]], dtype=np.float64))
            with self.assertRaises(ValueError):
                eg.Layout2d.from_coords(np.array([[bad_val, 0.0]], dtype=np.float32))
            with self.assertRaises(ValueError):
                eg.Layout2d.from_coords([[0.0, bad_val]])

        # Invalid coordinate element types
        with self.assertRaises(TypeError):
            eg.Layout2d.from_coords(np.array([[1, 2]], dtype=np.int64))
        with self.assertRaises(TypeError):
            eg.Layout2d.from_coords([[True, False]])
        with self.assertRaises(TypeError):
            eg.Layout2d.from_coords("not_coords")  # type: ignore[arg-type]

    def test_package_exports_and_docstrings(self) -> None:
        # CSR/SoA classes and functions are exposed on top-level `egraph`
        self.assertIs(eg.CsrGraph, eg._core.CsrGraph)
        self.assertIs(eg.Layout2d, eg._core.Layout2d)
        self.assertTrue((eg.CsrGraph.__doc__ or "").strip())
        self.assertTrue((eg.Layout2d.__doc__ or "").strip())
        sig = inspect.signature(eg.Layout2d.zeros)
        self.assertEqual(list(sig.parameters.keys()), ["n", "dtype"])

        # Every public class, method, and function must expose a valid inspect.signature
        for name in eg.__all__:
            obj = getattr(eg, name)
            if inspect.isclass(obj):
                if issubclass(obj, BaseException):
                    continue
                inspect.signature(obj)
                for attr_name in dir(obj):
                    if attr_name.startswith("__") and attr_name != "__new__":
                        continue
                    member = getattr(obj, attr_name)
                    if callable(member):
                        inspect.signature(member)
            elif callable(obj):
                inspect.signature(obj)

    def test_version_export_matches_metadata(self) -> None:
        meta_ver = importlib.metadata.version("egraph")
        self.assertEqual(eg.__version__, meta_ver)
        self.assertEqual(eg._core.__version__, meta_ver)
        self.assertIsInstance(eg.__version__, str)
        self.assertNotIn("__version__", eg.__all__)


if __name__ == "__main__":
    unittest.main()
