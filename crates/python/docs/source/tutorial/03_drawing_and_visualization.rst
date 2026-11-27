Drawing and Visualization
=========================

This tutorial covers coordinate buffers, geometric spaces, and integration with matplotlib.

Understanding Coordinate Spaces
-------------------------------

egraph v6 supports multiple coordinate containers and geometric spaces:

* **Euclidean 2D**: :class:`egraph.Layout2d` (zero-copy ``float64`` or ``float32`` SoA buffers)
* **Euclidean nD**: :class:`egraph.LayoutNd` (arbitrary-dimensional embeddings with ``.dims`` and :class:`egraph.ArenaOctree3d` for 3D Barnes-Hut repulsion across 11 $N$-D solvers)
* **Manifold 2D**: :class:`egraph.ManifoldLayout2d` (``"hyperbolic"``, ``"hyperbolic_lorentz"``, ``"spherical"``, ``"torus"`` across 6 manifold solvers)

Euclidean 2D Layout
-------------------

Creating a Layout
^^^^^^^^^^^^^^^^^

.. testcode:: python

    import egraph as eg

    graph = eg.CsrGraph.from_edges(5, [(i, i + 1) for i in range(4)])

    # Create 2D Euclidean layout with deterministic random initial positions
    layout = eg.Layout2d.init(graph.num_nodes, seed=42)
    print(f"Created 2D layout for {layout.num_nodes} nodes")

.. testoutput:: python

    Created 2D layout for 5 nodes

Accessing Coordinates
^^^^^^^^^^^^^^^^^^^^^

.. testcode:: python

    import egraph as eg

    layout = eg.Layout2d.init(3, seed=42)

    for i in range(3):
        print(f"Node {i}: ({layout.x(i):.2f}, {layout.y(i):.2f})")

.. testoutput:: python
    :options: +SKIP

    Node 0: (0.00, 0.00)
    Node 1: (1.23, 4.56)
    Node 2: (-2.34, 3.45)

Setting Coordinates & Zero-Copy NumPy Views
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. testcode:: python

    import egraph as eg

    layout = eg.Layout2d.zeros(3)
    layout.set_x(0, 0.0)
    layout.set_y(0, 0.0)
    layout.set_x(1, 1.0)
    layout.set_y(1, 0.0)
    layout.set_x(2, 0.5)
    layout.set_y(2, 1.0)

    view = layout.coords(copy=False)
    assert view.shape == (3, 2)
    print("Coordinates set manually")

.. testoutput:: python

    Coordinates set manually

Non-Euclidean Manifold Layouts
------------------------------

Use :class:`egraph.ManifoldLayout2d` for non-Euclidean geometries (``"hyperbolic"``,
``"hyperbolic_lorentz"``, ``"spherical"``, ``"torus"``) with polymorphic ``step(manifold)`` / ``run(manifold)`` across 6 manifold solvers (:class:`egraph.SgdSolver`, :class:`egraph.TsNetSolver`, :class:`egraph.GuMapSolver`, :class:`egraph.StressMajorizationSolver`, :class:`egraph.KamadaKawaiSolver`, :class:`egraph.FruchtermanReingoldSolver`).

.. testcode:: python

    import egraph as eg

    graph = eg.CsrGraph.from_edges(6, [(i, (i + 1) % 6) for i in range(6)])
    for kind in ("hyperbolic", "hyperbolic_lorentz", "spherical", "torus"):
        manifold = eg.ManifoldLayout2d.init(kind, graph.num_nodes, seed=42)
        sgd = eg.SgdSolver.full(graph, iterations=30, seed=42)
        sgd.run(manifold)
        assert manifold.coords().shape == (6, 2)
        assert manifold.kind == kind

    # Convert Lorentz hyperboloid coordinates to Poincaré disk & pan torus layouts
    lorentz = eg.ManifoldLayout2d.init("hyperbolic_lorentz", graph.num_nodes, seed=42)
    eg.SgdSolver.full(graph, iterations=30, seed=42).run(lorentz)
    poincare = lorentz.to_poincare_layout()
    poincare.rescale_geodesic(1.5)
    assert poincare.kind == "hyperbolic" and poincare.scale == 1.5

    torus = eg.ManifoldLayout2d.torus_spectral_phase_init(graph, seed=42)
    panned = torus.torus_auto_pan(graph)
    assert panned.coords().shape == (6, 2)
    print("Manifold layouts complete")

.. testoutput:: python

    Manifold layouts complete

3D Coordinate Buffers & Spatial Trees
-------------------------------------

.. testcode:: python

    import egraph as eg

    graph = eg.CsrGraph.from_edges(8, [(i, (i + 1) % 8) for i in range(8)])
    layout_3d = eg.LayoutNd.init(graph.num_nodes, dims=3, seed=42)
    eg.ForceAtlas2Solver(graph, iterations=20, seed=42).run(layout_3d)

    octree = eg.ArenaOctree3d()
    octree.build(layout_3d)
    forces = octree.compute_coulomb_repulsion(theta=0.5)
    assert layout_3d.dims == 3 and forces.shape == (8, 3)
    print("3D LayoutNd & ArenaOctree3d complete")

.. testoutput:: python

    3D LayoutNd & ArenaOctree3d complete

Visualization with Matplotlib
------------------------------

Basic Visualization
^^^^^^^^^^^^^^^^^^^

.. testcode:: python

    import networkx as nx
    import egraph as eg
    import matplotlib.pyplot as plt

    nx_graph = nx.karate_club_graph()
    graph, nodes = eg.from_networkx(nx_graph)

    layout = eg.Layout2d.init(graph.num_nodes, seed=42)
    eg.StressMajorizationSolver.from_graph(graph).run(layout)

    coords = layout.coords(copy=True)
    pos = {node: (float(coords[i, 0]), float(coords[i, 1]))
           for i, node in enumerate(nodes)}

    plt.figure(figsize=(10, 8))
    nx.draw(nx_graph, pos, node_color="lightblue", node_size=300, with_labels=True)
    plt.close()
    print("Visualization created")

.. testoutput:: python

    Visualization created

Customizing Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^

.. testcode:: python

    import networkx as nx
    import egraph as eg
    import matplotlib.pyplot as plt

    nx_graph = nx.karate_club_graph()
    graph, nodes = eg.from_networkx(nx_graph)

    layout = eg.Layout2d.init(graph.num_nodes, seed=42)
    eg.StressMajorizationSolver.from_graph(graph).run(layout)

    coords = layout.coords(copy=True)
    pos = {node: (float(coords[i, 0]), float(coords[i, 1]))
           for i, node in enumerate(nodes)}

    degrees = dict(nx_graph.degree())
    node_sizes = [degrees[node] * 50 for node in nx_graph.nodes()]

    communities = nx.community.greedy_modularity_communities(nx_graph)
    node_colors = []
    for node in nx_graph.nodes():
        for i, comm in enumerate(communities):
            if node in comm:
                node_colors.append(i)
                break

    plt.figure(figsize=(12, 10))
    nx.draw(
        nx_graph,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=plt.cm.Set3,
        with_labels=True,
        font_size=8,
        edge_color="gray",
        alpha=0.7,
    )
    plt.close()
    print("Styled visualization created")

.. testoutput:: python

    Styled visualization created

Next Steps
----------

* :doc:`../examples/index` - See complete v6 examples
* :doc:`../api/drawing` - Detailed coordinate buffer API reference
* :doc:`../getting_started/overview` - Review library capabilities
