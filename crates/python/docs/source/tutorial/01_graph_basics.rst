Graph Basics
============

This tutorial covers the fundamentals of creating and inspecting graphs in egraph v6.

Creating CSR Graphs
-------------------

egraph v6 uses a single zero-copy Compressed Sparse Row (CSR) graph representation,
:class:`egraph.CsrGraph`, for both undirected and directed graphs.

Creating an Undirected or Directed Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. testcode:: python

    import egraph as eg

    # Create an undirected graph from an edge list
    graph = eg.CsrGraph.from_edges(3, [(0, 1), (1, 2), (0, 2)])
    print(f"Undirected: {graph.num_nodes} nodes, {graph.num_edges} edges")

    # Create a directed graph
    digraph = eg.CsrGraph.from_edges(3, [(0, 1), (1, 2), (2, 0)], directed=True)
    print(f"Directed: {digraph.num_nodes} nodes, {digraph.num_edges} edges")

.. testoutput:: python

    Undirected: 3 nodes, 3 edges
    Directed: 3 nodes, 3 edges

Creating Weighted Graphs
^^^^^^^^^^^^^^^^^^^^^^^^

Edges can carry positive scalar weights for stress, shortest-path, and community algorithms:

.. testcode:: python

    import egraph as eg

    weighted = eg.CsrGraph.from_edges(
        3,
        [(0, 1), (1, 2), (0, 2)],
        weights=[1.0, 2.5, 1.5],
    )
    print(f"Weighted: {weighted.is_weighted}, edges={weighted.num_edges}")

.. testoutput:: python

    Weighted: True, edges=3

Inspecting Graph Topology
-------------------------

Adjacency, Degrees, and CSR Arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. testcode:: python

    import egraph as eg

    graph = eg.CsrGraph.from_edges(4, [(0, 1), (1, 2), (1, 3)])
    print(f"Degree of node 1: {graph.degree(1)}")
    print(f"Neighbors of node 1: {graph.neighbors(1).tolist()}")

.. testoutput:: python

    Degree of node 1: 3
    Neighbors of node 1: [0, 2, 3]

Largest Connected Component & Subgraphs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. testcode:: python

    import egraph as eg

    graph = eg.CsrGraph.from_edges(5, [(0, 1), (1, 2), (3, 4)])
    lcc, orig_ids = graph.largest_connected_component()
    print(f"LCC has {lcc.num_nodes} nodes, {lcc.num_edges} edges")

.. testoutput:: python

    LCC has 3 nodes, 2 edges

Working with NetworkX
---------------------

``egraph.interop`` integrates directly with NetworkX, igraph, and SciPy CSR matrices.

Converting from NetworkX
^^^^^^^^^^^^^^^^^^^^^^^^

.. testcode:: python

    import networkx as nx
    import egraph as eg

    nx_graph = nx.karate_club_graph()
    graph, nodes = eg.from_networkx(nx_graph)
    print(f"Converted graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

.. testoutput:: python

    Converted graph: 34 nodes, 78 edges

Using NetworkX Algorithms with egraph Layouts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can combine NetworkX analysis with egraph v6 layout solvers:

.. testcode:: python

    import networkx as nx
    import egraph as eg

    nx_graph = nx.karate_club_graph()
    communities = nx.community.greedy_modularity_communities(nx_graph)

    graph, _ = eg.from_networkx(nx_graph)
    layout = eg.Layout2d.init(graph.num_nodes, seed=42)
    eg.StressMajorizationSolver.from_graph(graph).run(layout)

    print(f"Found {len(communities)} communities")
    print(f"Layout computed for {graph.num_nodes} nodes")

.. testoutput:: python

    Found 3 communities
    Layout computed for 34 nodes

Built-in Datasets & CSV Parsing
-------------------------------

.. testcode:: python

    import egraph as eg

    karate = eg.load_dataset("karate")
    csv_graph = eg.parse_csr_csv("3,2\n0,1,1.5\n1,2,2.0\n")
    print(f"karate: {karate.num_nodes} nodes, csv weighted: {csv_graph.is_weighted}")

.. testoutput:: python

    karate: 34 nodes, csv weighted: True

Best Practices
--------------

1. **Use ``CsrGraph.from_edges`` or ``CsrGraph.from_csr``** for zero-copy graph construction.
2. **Use ``sanitize=True``** (the default) when loading raw edge lists that may contain duplicate edges or self-loops.
3. **Use ``directed=True``** only when edge direction matters (for DAG layering or directed stress/crossings).
4. **Leverage ``egraph.interop``** for NetworkX, igraph, and SciPy conversions.

Next Steps
----------

* :doc:`02_layout_algorithms` - Learn about different layout algorithms
* :doc:`03_drawing_and_visualization` - Explore coordinate buffers and visualization
* :doc:`../examples/index` - See more complex examples
