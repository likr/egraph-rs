Layout Algorithms
=================

This tutorial explores the different layout algorithms available in egraph v6 and when to use each one.

Understanding Layout Algorithms
--------------------------------

Layout algorithms compute positions for graph nodes to create readable visualizations.
All iterative solvers in egraph v6 share a uniform cooperative stepping interface
(``step(layout)``, ``run_steps(layout, steps)``, ``run(layout)``) over :class:`egraph.Layout2d`.

Stress Majorization
-------------------

Stress Majorization is an iterative algorithm that minimizes graph-theoretic stress, producing high-quality layouts.

**Best for**: Small to medium graphs where stress accuracy is important

.. testcode:: python

    import egraph as eg

    graph = eg.load_dataset("karate")
    layout = eg.Layout2d.init(graph.num_nodes, seed=42)

    solver = eg.StressMajorizationSolver.from_graph(graph)
    solver.run(layout)

    print(f"Layout computed for {graph.num_nodes} nodes")

.. testoutput:: python

    Layout computed for 34 nodes

Kamada-Kawai
------------

Kamada-Kawai is a spring-based algorithm that treats all pairs as springs at graph distance.

**Best for**: Small graphs and tree-like structures

.. testcode:: python

    import egraph as eg

    graph = eg.CsrGraph.from_edges(6, [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)])
    layout = eg.Layout2d.init(graph.num_nodes, seed=42)

    kk = eg.KamadaKawaiSolver.from_graph(graph)
    kk.run(layout)
    print(f"Kamada-Kawai layout for {graph.num_nodes} nodes complete")

.. testoutput:: python

    Kamada-Kawai layout for 6 nodes complete

SGD (Stochastic Gradient Descent)
----------------------------------

SGD algorithms are fast and scalable, making them ideal for medium to large graphs.

Full Stress SGD
^^^^^^^^^^^^^^^

Computes all-pairs shortest paths for accurate layouts on small to medium graphs.

.. testcode:: python

    import egraph as eg

    graph = eg.CsrGraph.from_edges(10, [(i, i + 1) for i in range(9)])
    layout = eg.Layout2d.init(graph.num_nodes, seed=42)

    sgd = eg.SgdSolver.full(graph, iterations=60, seed=42)
    sgd.run(layout)
    print("Full SGD layout complete")

.. testoutput:: python

    Full SGD layout complete

Regularized DoG Sparse Barrier SGD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Uses :math:`O(|E|)` Chebyshev DoG landmarks and 3-tier sparse barrier SGD for large graphs.

.. testcode:: python

    import egraph as eg

    graph = eg.load_dataset("lesmis")
    layout = eg.pivot_mds_init(graph, num_pivots=16, seed=42)

    sgd = eg.SgdSolver.sparse_barrier_dog(graph, iterations=60, seed=42)
    sgd.run(layout)
    print(f"Sparse barrier SGD layout for {graph.num_nodes} nodes complete")

.. testoutput:: python

    Sparse barrier SGD layout for 77 nodes complete

Omega (Spectral + Random-Pair SGD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Combines spectral coordinates from ``rd_mds`` with random-pair SGD.

.. testcode:: python

    import egraph as eg

    graph = eg.load_dataset("karate")
    layout = eg.Layout2d.init(graph.num_nodes, seed=42)

    embedding = eg.rd_mds(graph, dims=2, seed=42)
    sgd = eg.SgdSolver.omega(graph, embedding.coords(), k=20, iterations=60, seed=42)
    sgd.run(layout)
    print("Omega layout complete")

.. testoutput:: python

    Omega layout complete

MDS (Multidimensional Scaling)
-------------------------------

MDS algorithms preserve graph-theoretic distances in low-dimensional embeddings.

Classical MDS
^^^^^^^^^^^^^

Uses exact all-pairs shortest paths and eigendecomposition for small graphs.

.. testcode:: python

    import egraph as eg

    graph = eg.CsrGraph.from_edges(8, [(i, i + 1) for i in range(7)])
    layout = eg.classical_mds_2d(graph)
    assert layout.num_nodes == 8
    print("Classical MDS layout complete")

.. testoutput:: python

    Classical MDS layout complete

Pivot MDS
^^^^^^^^^

Scalable landmark-based MDS for fast initialization on larger graphs.

.. testcode:: python

    import egraph as eg

    graph = eg.load_dataset("lesmis")
    layout = eg.pivot_mds_init(graph, num_pivots=20, seed=42)
    assert layout.is_all_finite()
    print(f"Pivot MDS layout for {layout.num_nodes} nodes complete")

.. testoutput:: python

    Pivot MDS layout for 77 nodes complete

Customizing Edge Lengths
-------------------------

Weighted edges in ``CsrGraph`` are respected directly by shortest-path and stress solvers:

.. testcode:: python

    import egraph as eg

    graph = eg.CsrGraph.from_edges(
        5,
        [(0, 1), (1, 2), (2, 3), (3, 4)],
        weights=[1.0, 2.0, 1.0, 3.0],
    )
    layout = eg.Layout2d.init(graph.num_nodes, seed=42)
    eg.StressMajorizationSolver.from_graph(graph).run(layout)
    print("Custom edge lengths applied")

.. testoutput:: python

    Custom edge lengths applied

3D / N-Dimensional Layouts & Post-Layout Decluttering
-----------------------------------------------------

Use :class:`egraph.Layout3d` with ``target="layout3d"`` for Barnes-Hut 3D layouts (or :class:`egraph.LayoutNd` with ``target="layout_nd"`` for arbitrary $N$-D), and :class:`egraph.TwinDeclutterer` (a ``LayoutTransform`` using ``.apply(layout)``) to separate co-located or near-twin nodes after layout:

.. testcode:: python

    import egraph as eg

    graph = eg.load_dataset("karate")
    layout_3d = eg.Layout3d.init(graph.num_nodes, seed=42)
    eg.ForceAtlas2Solver(graph, target="layout3d", iterations=30, seed=42).run(layout_3d)
    assert layout_3d.coords(copy=False).shape == (graph.num_nodes, 3) and layout_3d.is_all_finite()

    layout_2d = eg.pivot_mds_init(graph, num_pivots=16, seed=42)
    eg.TsNetSolver.from_graph_row_normalized(
        graph,
        backend="auto",
        stage_iterations=(15, 15, 20),
        seed=42,
    ).run(layout_2d)
    eg.TwinDeclutterer(iterations=4).apply(layout_2d)
    assert layout_2d.is_all_finite()
    print("3D and decluttered 2D layouts complete")

.. testoutput:: python

    3D and decluttered 2D layouts complete

Next Steps
----------

* :doc:`03_drawing_and_visualization` - Learn about coordinate buffers and visualization
* :doc:`../examples/index` - See Track A / Track B and manifold examples
* :doc:`../api/layout` - Detailed API reference
