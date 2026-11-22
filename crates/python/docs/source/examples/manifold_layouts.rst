Non-Euclidean Manifold Layouts (``ManifoldLayout2d``)
=====================================================

This example demonstrates non-Euclidean layouts on the Poincare hyperbolic disk, spherical surface, and 2D periodic torus using :class:`egraph.ManifoldLayout2d` and :class:`egraph.SgdSolver`.

.. testcode:: python

    import egraph as eg

    graph = eg.CsrGraph.from_edges(
        8,
        [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 6), (3, 7)],
    )

    for kind in ("hyperbolic", "spherical", "torus"):
        manifold = eg.ManifoldLayout2d.zeros(kind, graph.num_nodes)
        sgd = eg.SgdSolver.full(graph, t_max=30, seed=42)
        sgd.run_manifold(manifold)
        coords = manifold.coords()
        d01 = manifold.geodesic_distance(0, 1)
        assert coords.shape == (8, 2) and d01 >= 0.0
        print(f"{kind}: shape={coords.shape}")

.. testoutput:: python

    hyperbolic: shape=(8, 2)
    spherical: shape=(8, 2)
    torus: shape=(8, 2)
