Migrating to egraph v6
======================

``egraph`` v6 (``6.0.0a4``) introduces a clean-break, strictly single-threaded, zero-copy architecture across Native Rust, Python, and WebAssembly.

For the full side-by-side Python and WebAssembly/TypeScript before/after guide, see ``docs/MIGRATION_V6.md`` in the repository (`MIGRATION_V6.md on GitHub <https://github.com/likr/egraph-rs/blob/master/docs/MIGRATION_V6.md>`_).

Summary of Key Changes
----------------------

1. **Host Object Weights Removed -> ``CsrGraph`` + ``egraph.interop`` + Host-Side Arrays**

   Replace ``Graph()`` / ``DiGraph()`` node/edge Python object payloads with ``CsrGraph.from_csr(...)``, ``CsrGraph.from_edges(...)``, or ``egraph.interop`` helpers (``from_scipy_csr``, ``from_networkx``, ``from_igraph``) while keeping labels and metadata in host-side Python lists or NumPy arrays.

2. **``OverwrapRemoval`` -> ``PrismOverlapRemoval``**

   Replace legacy ``OverwrapRemoval`` with :class:`egraph.PrismOverlapRemoval` (``from_radii`` or rectangular ``half_widths`` / ``half_heights``) and evaluate fidelity with :func:`egraph.prism_distortion`.

3. **``TsNet`` / ``BhTsNet`` / ``FitTsNet`` / ``LTsNet`` -> ``TsNetSolver``**

   Replace the four legacy tsNET classes with the unified :class:`egraph.TsNetSolver` (``TsNetSolver.pure_2hop``, ``from_knn``, ``from_probabilities``, ``from_graph_partial_bfs``), supporting ``backend="fft" | "linear_fft" | "barnes_hut" | "exact"`` and cooperative ``step()`` / ``run_steps()`` / ``run()``.

4. **``stress`` (``NS``) -> Scale-Normalized Stress (``SNS``) / ``evaluate_metrics``**

   Replace scale-sensitive ``stress(drawing, dm)`` with :func:`egraph.scale_normalized_stress`, :func:`egraph.sampled_scale_normalized_stress`, or selective :func:`egraph.evaluate_metrics`.

5. **Non-Euclidean & Unified Force Models**

   Replace ``DrawingEuclidean2d`` with :class:`egraph.Layout2d`, and replace ``DrawingHyperbolic2d`` / ``DrawingSpherical2d`` / ``DrawingTorus2d`` with :class:`egraph.ManifoldLayout2d` + :meth:`egraph.SgdSolver.run_manifold`. ``InfoMap`` is excluded from the v6 API (``egraph-clustering``).
