Force-Directed Edge Bundling (``FdebSolver``)
=============================================

This example demonstrates cooperative Force-Directed Edge Bundling using :class:`egraph.FdebSolver`, which accepts a ``CsrGraph`` in its constructor and takes a ``Layout2d`` or ``LayoutNd`` in ``step()`` / ``run_steps()`` / ``run()``, outputting flat ``edge_offsets()`` and ``points()`` NumPy arrays.

.. testcode:: python

    import egraph as eg

    # Create a bipartite-like graph with crossing edges
    edges = [(0, 4), (0, 5), (1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)]
    graph = eg.CsrGraph.from_edges(6, edges)
    layout = eg.Layout2d.from_coords(
        [
            [-1.0, -1.5],
            [-1.0, -0.5],
            [-1.0, 0.5],
            [-1.0, 1.5],
            [1.0, -0.5],
            [1.0, 0.5],
        ]
    )

    fdeb = eg.FdebSolver(graph, cycles=3, steps_per_cycle=15)
    fdeb.run(layout)

    offsets = fdeb.edge_offsets()
    points = fdeb.points()
    assert len(offsets) == fdeb.num_edges + 1
    assert points.ndim == 2 and points.shape[1] == 2
    print("Bundled edges:", fdeb.num_edges)

.. testoutput:: python

    Bundled edges: 8
