Force-Directed Edge Bundling (``FdebSolver``)
=============================================

This example demonstrates cooperative Force-Directed Edge Bundling using :class:`egraph.EdgePolylines` and :class:`egraph.FdebSolver`. Construct ``EdgePolylines`` via ``EdgePolylines.straight_line_from_graph(graph, layout)``, pass it to ``FdebSolver(polylines, ...)``, and step in place via ``step(polylines)`` / ``run_steps(polylines, k)`` / ``run(polylines)``.

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

    polylines = eg.EdgePolylines.straight_line_from_graph(graph, layout)
    fdeb = eg.FdebSolver(polylines, cycles=3, steps_per_cycle=15)
    fdeb.run(polylines)

    offsets = polylines.edge_offsets()
    points = polylines.points()
    assert len(offsets) == fdeb.num_edges + 1
    assert points.ndim == 2 and points.shape[1] == 2
    print("Bundled edges:", fdeb.num_edges)

.. testoutput:: python

    Bundled edges: 8
