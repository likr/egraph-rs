Examples
========

This section provides practical examples demonstrating ``egraph`` v6 features.

.. toctree::
   :maxdepth: 2

   track_a_tsnet
   track_b_sparse_barrier
   unified_force_models
   prism_overlap_removal
   fdeb
   manifold_layouts

Overview
--------

Each example is self-contained and verified by Sphinx ``doctest``.

v6 Solvers & Workflows
----------------------

* :doc:`track_a_tsnet` - Track A Pure 1+2-Hop Adaptive tsNET (``TsNetSolver.pure_2hop``)
* :doc:`track_b_sparse_barrier` - Track B Regularized DoG 3-Tier Sparse Barrier SGD, ``GuMapSolver``, and ``SpatialIntruderRefiner``
* :doc:`unified_force_models` - ``TFdpSolver``, ``IbUmapSolver``, ``SgdSolver.snap_t_fdp``, ``SgdSolver.pacmap_graph``, ``ForceAtlas2Solver``, and ``YifanHuSolver``
* :doc:`prism_overlap_removal` - $O(N \log N)$ Delaunay PRISM overlap removal and ``prism_distortion``
* :doc:`fdeb` - Cooperative Force-Directed Edge Bundling (``FdebSolver``)
* :doc:`manifold_layouts` - Non-Euclidean ``ManifoldLayout2d`` (Hyperbolic, Spherical, Torus)

Quick Example
-------------

.. testcode:: python

    import egraph as eg

    # Create a 4x4 grid graph via CsrGraph.from_edges
    edges = []
    for r in range(4):
        for c in range(4):
            u = r * 4 + c
            if c + 1 < 4:
                edges.append((u, u + 1))
            if r + 1 < 4:
                edges.append((u, u + 4))

    graph = eg.CsrGraph.from_edges(16, edges)
    layout = eg.pivot_mds_init(graph, num_pivots=8, seed=42)
    eg.TsNetSolver.pure_2hop(
        graph,
        iterations_stage1=20,
        iterations_stage2=20,
        iterations_stage3=30,
        seed=42,
    ).run(layout)

    report = eg.evaluate_metrics(graph, layout, ["sns", "np2", "crossing_number"])
    assert layout.is_all_finite()
    assert 0.0 <= report["neighborhood_preservation_2hop"] <= 1.0
    print("Quick example complete")

.. testoutput:: python

    Quick example complete
