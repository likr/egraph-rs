Quick Start
===========

This guide shows how to build a ``CsrGraph``, run v6 layout solvers, inspect zero-copy NumPy coordinates, and evaluate layout quality metrics.

Your First v6 Graph Layout (``CsrGraph`` + ``TsNetSolver.pure_2hop``)
----------------------------------------------------------------------

Here is a minimal example creating a cycle graph directly from CSR arrays, initializing with normalized PivotMDS, optimizing with Track A ``TsNetSolver.pure_2hop``, and evaluating quality metrics:

.. testcode:: python

    import numpy as np
    import egraph as eg

    # Create a 6-node cycle graph (0 - 1 - 2 - 3 - 4 - 5 - 0) from CSR arrays
    indptr = np.array([0, 2, 4, 6, 8, 10, 12], dtype=np.uint32)
    indices = np.array([1, 5, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0], dtype=np.uint32)
    graph = eg.CsrGraph.from_csr(indptr, indices)

    # Initialize 2D layout using normalized PivotMDS
    layout = eg.pivot_mds_init(graph, num_pivots=6, seed=42)

    # Optimize with Track A Pure 1+2-Hop Adaptive tsNET
    solver = eg.TsNetSolver.pure_2hop(
        graph,
        backend="fft",
        iterations_stage1=30,
        iterations_stage2=30,
        iterations_stage3=40,
        seed=42,
    )
    solver.run(layout)

    # Access coordinates via a zero-copy NumPy view (shape: (6, 2))
    coords = layout.coords(copy=False)
    assert coords.shape == (6, 2)
    assert layout.is_all_finite()

    # Selectively evaluate SNS, NP2, and crossing number
    report = eg.evaluate_metrics(graph, layout, ["sns", "np2", "crossing_number"])
    print("Metric keys:", sorted(report.keys()))

.. testoutput:: python

    Metric keys: ['crossing_number', 'neighborhood_preservation_2hop', 'scale_normalized_stress']

Using Track B (``SgdSolver.sparse_barrier_dog``) and ``TFdpSolver``
-------------------------------------------------------------------

You can switch solvers on the same ``CsrGraph`` and ``Layout2d`` buffer, including ``dtype="float32"`` for large-scale memory efficiency:

.. testcode:: python

    import egraph as eg

    edges = [(i, (i + 1) % 10) for i in range(10)] + [(0, 5), (2, 7)]
    graph = eg.CsrGraph.from_edges(10, edges, directed=False, sanitize=True)

    # Track B: Regularized Chebyshev DoG + 3-tier Sparse Barrier SGD
    layout_b = eg.pivot_mds_init(graph, num_pivots=6, seed=42)
    barrier = eg.SgdSolver.sparse_barrier_dog(graph, seed=42)
    barrier.run(layout_b)
    stats = barrier.dog_stats()
    assert stats is not None and layout_b.is_all_finite()

    # Bounded Student-t FDP with ibFFT repulsion (float32)
    layout_f32 = eg.pivot_mds_init(graph, num_pivots=6, seed=42, dtype="float32")
    tfdp = eg.TFdpSolver(graph, backend="ibfft", total_iterations=50, seed=42, dtype="float32")
    tfdp.run(layout_f32)
    assert layout_f32.dtype == "float32" and layout_f32.is_all_finite()
    print("Track B & t-FDP complete")

.. testoutput:: python

    Track B & t-FDP complete

Working with NetworkX and SciPy CSR (``egraph.interop``)
--------------------------------------------------------

In v6, use ``egraph.interop`` to convert NetworkX graphs, SciPy CSR matrices, or igraph objects directly into ``CsrGraph`` without storing Python objects in Rust:

.. testcode:: python

    import networkx as nx
    import scipy.sparse as sp
    import egraph as eg
    from egraph.interop import from_networkx, from_scipy_csr

    # Convert a NetworkX graph to CsrGraph
    nx_graph = nx.karate_club_graph()
    graph, nodes = from_networkx(nx_graph)
    assert graph.num_nodes == len(nodes) == 34

    # Or convert a SciPy CSR adjacency matrix directly
    adj = sp.csr_matrix(nx.to_scipy_sparse_array(nx_graph, format="csr", dtype=float))
    graph_sp = from_scipy_csr(adj)
    assert graph.num_edges == graph_sp.num_edges

    layout = eg.pivot_mds_init(graph, num_pivots=16, seed=42)
    eg.TsNetSolver.pure_2hop(
        graph,
        iterations_stage1=30,
        iterations_stage2=30,
        iterations_stage3=40,
        seed=42,
    ).run(layout)

    coords = layout.coords(copy=False)
    pos = {node: (float(coords[i, 0]), float(coords[i, 1])) for i, node in enumerate(nodes)}
    report = eg.evaluate_metrics(graph, layout, ["sns", "np2"])
    print("Interop nodes:", len(pos), "keys:", sorted(report.keys()))

.. testoutput:: python

    Interop nodes: 34 keys: ['neighborhood_preservation_2hop', 'scale_normalized_stress']

Cooperative Stepping
--------------------

Every v6 solver exposes ``step(layout)`` and ``run_steps(layout, steps)`` for progress tracking or incremental updates:

.. testcode:: python

    import egraph as eg

    graph = eg.CsrGraph.from_edges(8, [(i, (i + 1) % 8) for i in range(8)])
    layout = eg.pivot_mds_init(graph, num_pivots=4, seed=42)
    fa2 = eg.ForceAtlas2Solver(graph, iterations=20, seed=42)

    fa2.run_steps(layout, 10)
    assert fa2.iteration == 10 and not fa2.is_finished
    fa2.run(layout)
    assert fa2.is_finished
    print("Finished iterations:", fa2.iteration)

.. testoutput:: python

    Finished iterations: 20

Next Steps
----------

* :doc:`overview` - Review v6 architecture and features
* :doc:`migration_v6` - Migrate from v5 / legacy alpha APIs
* :doc:`../examples/index` - Explore Track A, Track B, PRISM, FDEB, and manifold examples
* :doc:`../api/index` - Full API reference
