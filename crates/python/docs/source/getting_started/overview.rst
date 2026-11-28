Overview
========

**egraph** is a high-performance, strictly single-threaded Python library for graph visualization and layout algorithms, powered by a pure CPU Rust core.

What is egraph v6?
------------------

egraph v6 provides a unified, zero-copy toolkit for:

* **Zero-copy CSR & SoA structures** - ``CsrGraph`` (``u32`` CSR topology + optional ``float64``/``float32`` weights), ``Layout2d``, ``Layout3d``, ``LayoutNd`` (``.dims``), ``ManifoldLayout2d``, and ``EdgePolylines``
* **3D / N-D spatial engine** - ``Layout3d``, ``ArenaOctree3d``, ``DynamicKdTree``, explicit ``target="layout3d" | "layout_nd"`` solvers, and $N$-D quality metrics
* **Non-Euclidean geometries** - Poincaré disk (``"hyperbolic"``), Lorentz hyperboloid (``"hyperbolic_lorentz"`` + ``to_poincare_layout()``), spherical surface (``"spherical"``), and 2D torus (``"torus"`` + ``torus_spectral_phase_init`` / ``torus_auto_pan``) with ``ManifoldMetric`` and ``target="manifold2d"`` solvers
* **Ecosystem interoperability** - ``egraph.interop`` conversions for SciPy CSR matrices, NetworkX graphs, and python-igraph (including ``to_igraph`` and 3D/$N$-D/manifold ``to_networkx``)
* **SOTA & separated layout solvers** - Track A ``TsNetSolver`` (``pure_2hop``, ``from_graph_row_normalized``, ``backend="auto"``), Track B ``SgdSolver.sparse_barrier_dog`` & ``GuMapSolver``, ``SpatialIntruderRefiner``, ``TFdpSolver``, ``IbUmapSolver``, separated embedding classes (``Sgd2Solver``, ``PacmapGraphSolver``, ``LargeVisSolver``, ``DrGraphSolver``, ``GraphCneSolver``, ``SnapTFdpSolver``), ``ForceAtlas2Solver``, ``YifanHuSolver``, ``LayoutGenerator`` classes (``ClassicalMds``, ``PivotMds``, ``Hde``, ``RdMds``, ``KorenSpectral``, ``CoarseBackbone``, ``SfcLayout``), ``PrismOverlapRemoval``, ``LayoutTransform.apply()`` (``PfsPrimeOverlapRemoval``, ``TwinDeclutterer``), ``transitive_reduction``, and ``EdgePolylines`` + ``FdebSolver(polylines)``
* **Community detection & datasets** - 5 typed zero-copy Result classes (``LouvainResult``, ``LeidenResult``, ``KMeansResult``, ``LabelPropagationResult``, ``SpectralClusteringResult``), ``list_datasets``, and 7 synthetic generators
* **Selective quality evaluation (``MetricReport``) & ``EgraphError``** - ``evaluate_metrics`` with closed-form and sampled Scale-Normalized Stress (``SNS``), ``geodesic_distortion``, sweep-line & manifold crossings, ``StaticKdTree`` ``NP1``/``NP2``, ``PRISM`` distortion, ``G-CAF``, ``NOR``, ``sprawlter_nn``, Eades et al. (2017) unweighted ``S_RNG``/``S_GG`` (``drng_preservation``/``dgg_preservation``), and structured ``EgraphError(ValueError)`` exceptions

Key Features
------------

Zero-Copy & Single-Threaded Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

egraph v6 eliminates per-node host object allocations and threading non-determinism:

* **Flat CSR / SoA buffers**: ``CsrGraph`` stores compact ``u32`` arrays and ``Layout2d.coords(copy=False)`` / ``Layout3d.coords(copy=False)`` / ``LayoutNd.coords(copy=False)`` expose direct NumPy views
* **Target-checked cooperative stepping**: All iterative solvers implement ``step(target)``, ``run_steps(target, k)``, and ``run(target)`` with explicit constructor ``target="layout2d" | "layout3d" | "layout_nd" | "manifold2d"`` validation (and ``EdgePolylines`` for ``FdebSolver``)
* **Deterministic RNG**: Seeded execution (``seed=42`` default) ensures reproducible layouts across Native Rust, Python, WebAssembly, and ``egraph-cli``

Layout Solvers
^^^^^^^^^^^^^^

* **Track A (``TsNetSolver``)**: Pure 1+2-hop CSR + node-wise adaptive perplexity (``pure_2hop``), direct row-normalized ``A1 + w2 * A2`` affinity (``from_graph_row_normalized`` with ``TwoStepRandomWalk``), ``from_knn``, ``from_probabilities``, and ``from_probabilities_with_stages`` (``TsNetStage``) across ``"auto"``, ``"fit_fft"``, ``"linear_fft"``, ``"barnes_hut"``, and ``"exact"`` repulsion backends
* **Track B (``SgdSolver`` / ``GuMapSolver`` / ``SpatialIntruderRefiner``)**: Regularized Chebyshev DoG landmarks (``ChebyshevDogKernel``, ``DogVaradhanLandmarks``), 3-tier sparse barrier SGD, GuMap, and 2D ``StaticKdTree`` spatial intruder refinement
* **Separated Embedding, Force & Multilevel Models**: ``TFdpSolver``, ``IbUmapSolver``, ``Sgd2Solver``, ``PacmapGraphSolver``, ``LargeVisSolver``, ``DrGraphSolver``, ``GraphCneSolver``, ``SnapTFdpSolver``, ``SgdSolver.ortmann_sparse_stress``, ``MaxentStressSolver``, ``GripSolver``, ``GemSolver``, ``SelLayoutSolver``, ``SfcLayout``, ``CoarseBackbone``, ``BackboneSolver``, ``GroupInABoxSolver``, ``ClusterRefineSolver``, ``ForceAtlas2Solver``, ``YifanHuSolver``
* **Classical, Overlap, Routing & Constraints**: ``StressMajorizationSolver``, ``KamadaKawaiSolver``, ``ClassicalMds``, ``PivotMds``, ``Hde``, ``RdMds``, ``KorenSpectral``, ``PrismOverlapRemoval``, ``ForbidOverlapRemoval`` (default ``"sparse_forbid"``), ``GTreeOverlapRemoval``, ``PfsPrimeOverlapRemoval``, ``TwinDeclutterer`` (``.apply(layout)``), ``declutter_twins_2d``, ``ScreenedPoissonField2d`` / ``ScreenedPoissonRefiner``, ``transitive_reduction``, ``SeparationConstraint`` projections, DAG layering, and ``EdgePolylines`` + ``FdebSolver(polylines)`` edge bundling

Quality Metrics
^^^^^^^^^^^^^^^

Evaluate layouts selectively without wasted computation using ``evaluate_metrics`` (returning :class:`egraph.MetricReport`):

* **Scale-Normalized Stress (``SNS``) & Geodesic Distortion** - Optimal-scale closed-form and $O(kN)$ pivot-sampled stress plus ``geodesic_distortion`` on manifolds
* **Edge Crossings** - Bentley-Ottmann sweep-line exact count, crossing angle, torus, spherical, and hyperbolic geodesic crossings, and sampled estimation
* **Neighborhood Preservation** - ``StaticKdTree``-accelerated 1-hop (``NP1``) and 2-hop (``NP2``)
* **Proximity, Cluster & Overlap Fidelity** - ``prism_distortion``, ``layout_degeneracy``, ``drng_preservation``, ``dgg_preservation``, ``gabriel_graph_property``, ``cluster_area_faithfulness``, ``landmark_stratified_rta``, ``adjusted_patch_nig``, ``nor_metrics``, ``sprawlter_nn``, and cluster separation baselines

Next Steps
----------

* :doc:`installation` - Install egraph and optional extras
* :doc:`quickstart` - Build and run your first v6 graph layout
* :doc:`migration_v6` - Migrate from v5 / legacy alpha APIs
* :doc:`../examples/index` - Explore complete v6 examples and interactive Colab notebooks
