Non-Euclidean Manifold Layouts (``ManifoldLayout2d``)
=====================================================

This example demonstrates non-Euclidean layouts on the Poincaré hyperbolic disk (``"hyperbolic"``), Lorentz hyperboloid (``"hyperbolic_lorentz"``), spherical surface (``"spherical"``), and 2D periodic torus (``"torus"``) using :class:`egraph.ManifoldLayout2d`, polymorphic ``run(manifold)`` solvers, ``ManifoldMetric`` rescaling, torus phase initialization/auto-panning, and geodesic metrics/crossings.

Hyperbolic, Lorentz, Spherical & Torus Optimization
---------------------------------------------------

.. testcode:: python

    import egraph as eg

    graph = eg.CsrGraph.from_edges(
        8,
        [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 6), (3, 7)],
    )

    for kind in ("hyperbolic", "hyperbolic_lorentz", "spherical", "torus"):
        manifold = eg.ManifoldLayout2d.init(kind, graph.num_nodes, seed=42)
        sgd = eg.SgdSolver.full(graph, target="manifold2d", iterations=30, seed=42)
        sgd.run(manifold)
        coords = manifold.coords()
        d01 = manifold.geodesic_distance(0, 1)
        assert coords.shape == (8, 2) and d01 >= 0.0
        print(f"{kind}: shape={coords.shape}")

.. testoutput:: python

    hyperbolic: shape=(8, 2)
    hyperbolic_lorentz: shape=(8, 2)
    spherical: shape=(8, 2)
    torus: shape=(8, 2)

Lorentz-to-Poincaré Conversion, ``ManifoldMetric``, Torus Auto-Pan & Geodesic Crossings
---------------------------------------------------------------------------------------

.. testcode:: python

    import egraph as eg

    # Hyperbolic tree on Lorentz hyperboloid -> Poincaré disk + geodesic rescaling
    tree = eg.generate_hyperbolic_tree(branching_factor=2, depth=3)
    lorentz = eg.ManifoldLayout2d.init("hyperbolic_lorentz", tree.num_nodes, seed=42)
    eg.SgdSolver.full(tree, target="manifold2d", iterations=40, seed=42).run(lorentz)
    poincare = lorentz.to_poincare_layout()
    poincare.rescale_geodesic(1.5)
    assert poincare.kind == "hyperbolic" and poincare.scale == 1.5
    gd = eg.geodesic_distortion(tree, poincare)
    hyp_crossings = eg.hyperbolic_crossing_number(tree, poincare)
    assert gd >= 0.0 and hyp_crossings >= 0

    # Spherical geodesic layout & spherical crossings
    ico = eg.generate_spherical_icosphere(subdivisions=1)
    sphere = eg.ManifoldLayout2d.init("spherical", ico.num_nodes, seed=42)
    eg.StressMajorizationSolver(ico, target="manifold2d", max_iterations=25).run(sphere)
    sph_crossings = eg.spherical_crossing_number(ico, sphere)
    assert sph_crossings >= 0

    # Torus spectral phase init, solver optimization, and optimal boundary auto-pan
    torus_graph = eg.generate_torus_grid(4, 4)
    torus = eg.ManifoldLayout2d.torus_spectral_phase_init(torus_graph, seed=42)
    eg.SgdSolver.full(torus_graph, target="manifold2d", iterations=30, seed=42).run(torus)
    panned = torus.torus_auto_pan(torus_graph)
    tor_crossings = eg.torus_crossing_number(torus_graph, panned)
    assert panned.coords().shape == (16, 2) and tor_crossings >= 0
    print("Manifold geodesics and crossings complete")

.. testoutput:: python

    Manifold geodesics and crossings complete
