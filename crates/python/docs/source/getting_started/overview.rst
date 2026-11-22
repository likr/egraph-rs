Overview
========

**egraph** is a high-performance, strictly single-threaded Python library for graph visualization and layout algorithms, powered by a pure CPU Rust core.

What is egraph v6?
------------------

egraph v6 provides a unified, zero-copy toolkit for:

* **Zero-copy CSR & SoA structures** - ``CsrGraph`` (``u32`` CSR topology + numeric weights), ``Layout2d`` (``float64`` default and ``float32``), ``LayoutNd``, and ``ManifoldLayout2d``
* **Ecosystem interoperability** - ``egraph.interop`` conversions for SciPy CSR matrices, NetworkX graphs, and igraph
* **SOTA & unified layout solvers** - Track A ``TsNetSolver``, Track B ``SgdSolver.sparse_barrier_dog`` & ``GuMapSolver``, ``SpatialIntruderRefiner``, ``TFdpSolver``, ``IbUmapSolver``, ``SgdSolver.snap_t_fdp``, ``SgdSolver.pacmap_graph``, ``ForceAtlas2Solver``, ``YifanHuSolver``, ``PrismOverlapRemoval``, and ``FdebSolver``
* **Non-Euclidean geometries** - Poincare hyperbolic disk, spherical surface, and 2D torus via ``ManifoldLayout2d``
* **Selective quality evaluation** - ``evaluate_metrics`` with closed-form and sampled Scale-Normalized Stress (``SNS``), sweep-line crossings, ``StaticKdTree`` ``NP1``/``NP2``, ``PRISM`` distortion, and ``dRNG``/``dGG``

Key Features
------------

Zero-Copy & Single-Threaded Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

egraph v6 eliminates per-node host object allocations and threading non-determinism:

* **Flat CSR / SoA buffers**: ``CsrGraph`` stores compact ``u32`` arrays and ``Layout2d.coords(copy=False)`` exposes a direct NumPy view
* **Cooperative stepping**: All iterative solvers implement ``step()``, ``run_steps(layout, k)``, and ``run(layout)``
* **Deterministic RNG**: Seeded execution (``seed=42`` default) ensures reproducible layouts across Native Rust, Python, and WebAssembly

Layout Solvers
^^^^^^^^^^^^^^

* **Track A (``TsNetSolver``)**: Pure 1+2-hop CSR + node-wise adaptive perplexity (``pure_2hop``), custom ``from_knn``, and ``from_probabilities`` with ``ibFFT``, Barnes-Hut, or Exact repulsion
* **Track B (``SgdSolver`` / ``GuMapSolver`` / ``SpatialIntruderRefiner``)**: Regularized Chebyshev DoG landmarks, 3-tier sparse barrier SGD, GuMap, and 2D ``StaticKdTree`` spatial intruder refinement
* **Unified Force & Multilevel Models**: ``TFdpSolver``, ``IbUmapSolver``, ``SgdSolver.snap_t_fdp``, ``SgdSolver.pacmap_graph``, ``SgdSolver.sgd2``, ``ForceAtlas2Solver``, ``YifanHuSolver``
* **Classical & Post-Processing**: ``StressMajorizationSolver``, ``KamadaKawaiSolver``, $O(N \log N)$ Delaunay ``PrismOverlapRemoval``, ``SeparationConstraint`` projections, DAG layering, and ``FdebSolver`` edge bundling

Quality Metrics
^^^^^^^^^^^^^^^

Evaluate layouts selectively without wasted computation using ``evaluate_metrics``:

* **Scale-Normalized Stress (``SNS``)** - Optimal-scale closed-form and $O(kN)$ pivot-sampled variants
* **Edge Crossings** - Bentley-Ottmann sweep-line exact count, crossing angle, torus crossings, and sampled estimation
* **Neighborhood Preservation** - ``StaticKdTree``-accelerated 1-hop (``NP1``) and 2-hop (``NP2``)
* **Proximity & Overlap Fidelity** - ``prism_distortion``, ``drng_preservation``, ``dgg_preservation``, ``gabriel_graph_property_v6``, ``node_resolution_v6``, ``aspect_ratio_v6``, ``angular_resolution_v6``, ``ideal_edge_lengths_v6``

Next Steps
----------

* :doc:`installation` - Install egraph and optional extras
* :doc:`quickstart` - Build and run your first v6 graph layout
* :doc:`migration_v6` - Migrate from v5 / legacy alpha APIs
* :doc:`../examples/index` - Explore complete v6 examples
