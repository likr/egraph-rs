Migrating to egraph v6
======================

``egraph`` v6 (``6.0.0a9``) introduces a clean-break, strictly single-threaded, zero-copy architecture across Native Rust, Python, and WebAssembly.

For the full side-by-side Python and WebAssembly/TypeScript before/after guide—including the ``alpha.8 -> alpha.9`` clean-break remediation summary—see ``docs/MIGRATION_V6.md`` in the repository (`MIGRATION_V6.md on GitHub <https://github.com/likr/egraph-rs/blob/master/docs/MIGRATION_V6.md>`_).

Summary of Key Changes
----------------------

1. **Host Object Weights Removed -> ``CsrGraph`` + ``egraph.interop`` + Host-Side Arrays**

   Replace ``Graph()`` / ``DiGraph()`` node/edge Python object payloads with ``CsrGraph.from_csr(...)``, ``CsrGraph.from_edges(...)``, or ``egraph.interop`` helpers (``from_scipy_csr``, ``to_scipy_csr``, ``from_networkx``, ``to_networkx``, ``from_igraph``, ``to_igraph``) while keeping labels and metadata in host-side Python lists or NumPy arrays.

2. **``OverwrapRemoval`` -> ``PrismOverlapRemoval``, ``ForbidOverlapRemoval`` & ``TwinDeclutterer``**

   Replace legacy ``OverwrapRemoval`` with :class:`egraph.PrismOverlapRemoval` (``from_radii`` or rectangular ``half_widths`` / ``half_heights``), :class:`egraph.ForbidOverlapRemoval` (default ``variant="sparse_forbid"``), and :class:`egraph.TwinDeclutterer`, and evaluate fidelity with :func:`egraph.prism_distortion`.

3. **``TsNet`` / ``BhTsNet`` / ``FitTsNet`` / ``LTsNet`` -> ``TsNetSolver``**

   Replace the four legacy tsNET classes with the unified :class:`egraph.TsNetSolver` (``TsNetSolver.pure_2hop``, ``from_graph_row_normalized``, ``from_knn``, ``from_probabilities``, ``from_probabilities_with_stages``, ``from_graph_partial_bfs``), supporting ``backend="auto" | "fft" | "linear_fft" | "barnes_hut" | "exact"`` and cooperative polymorphic ``step(layout)`` / ``run_steps(layout, k)`` / ``run(layout)``.

4. **``stress`` (``NS``) -> Scale-Normalized Stress (``SNS``) / ``MetricReport``**

   Replace scale-sensitive ``stress(drawing, dm)`` with :func:`egraph.scale_normalized_stress`, :func:`egraph.sampled_scale_normalized_stress`, or selective :func:`egraph.evaluate_metrics` returning :class:`egraph.MetricReport`.

5. **Iteration Arguments, ``labels`` & Typed Cluster Results**

   Fixed-iteration solvers standardize on ``iterations`` (replacing ``t_max``, ``total_iterations``, ``n_epochs``, ``rounds``), convergent solvers standardize on ``max_iterations`` (replacing ``max_rounds``), and multi-stage solvers use structured names (``stage_iterations``, ``phase_iterations``, ``steps_per_cycle``). Community detection functions accept/return unified ``labels`` via 5 typed zero-copy Result classes (:class:`egraph.LouvainResult`, :class:`egraph.LeidenResult`, :class:`egraph.KMeansResult`, :class:`egraph.LabelPropagationResult`, :class:`egraph.SpectralClusteringResult`). ``InfoMap`` is excluded from v6.

6. **3D / N-D & Non-Euclidean Manifolds**

   Replace ``DrawingEuclidean2d`` with :class:`egraph.Layout2d`, :class:`egraph.Layout3d`, or :class:`egraph.LayoutNd` (with ``.dims`` and :class:`egraph.ArenaOctree3d` for 3D Barnes-Hut repulsion), and replace ``DrawingHyperbolic2d`` / ``DrawingSpherical2d`` / ``DrawingTorus2d`` with :class:`egraph.ManifoldLayout2d` (``"hyperbolic"``, ``"hyperbolic_lorentz"``, ``"spherical"``, ``"torus"``).

7. **API Freeze Contracts (``target=``, ``EdgePolylines``, Separated Classes, ``EgraphError``)**

   Pass explicit ``target="layout2d" | "layout3d" | "layout_nd" | "manifold2d"`` when constructing solvers for non-2D spaces, construct :class:`egraph.EdgePolylines` for :class:`egraph.FdebSolver`, use separated SGD classes (:class:`egraph.Sgd2Solver`, :class:`egraph.PacmapGraphSolver`, :class:`egraph.LargeVisSolver`, :class:`egraph.DrGraphSolver`, :class:`egraph.GraphCneSolver`, :class:`egraph.SnapTFdpSolver`), ``LayoutGenerator.generate()`` (:class:`egraph.PivotMds`, :class:`egraph.ClassicalMds`, :class:`egraph.Hde`, :class:`egraph.RdMds`, :class:`egraph.KorenSpectral`, :class:`egraph.CoarseBackbone`, :class:`egraph.SfcLayout`), ``LayoutTransform.apply()`` (:class:`egraph.PfsPrimeOverlapRemoval`, :class:`egraph.TwinDeclutterer`), keyword-only ``*`` parameters, frozen Result objects (:class:`egraph.KnnBuildResult`, :class:`egraph.RhghLandmarksResult`, etc.), and the :class:`egraph.EgraphError` exception hierarchy.
