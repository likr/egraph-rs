Graph Basics
============

This tutorial covers the fundamentals of creating and manipulating graphs in egraph.

Creating Graphs
---------------

egraph provides two main graph types: **Graph** (undirected) and **DiGraph** (directed).

Creating an Empty Graph
^^^^^^^^^^^^^^^^^^^^^^^

.. testcode:: python

    import egraph as eg

    # Create an undirected graph
    graph = eg.Graph()
    
    # Create a directed graph
    digraph = eg.DiGraph()

Adding Nodes
^^^^^^^^^^^^

Nodes can be added with associated data:

.. testcode:: python

    import egraph as eg

    graph = eg.Graph()
    
    # Add nodes with data
    node0 = graph.add_node("Alice")
    node1 = graph.add_node("Bob")
    node2 = graph.add_node("Charlie")
    
    print(f"Added {graph.node_count()} nodes")

.. testoutput:: python

    Added 3 nodes

The `add_node()` method returns a node index that you use to reference the node later.

Adding Edges
^^^^^^^^^^^^

Edges connect nodes and can also carry data:

.. testcode:: python

    import egraph as eg

    graph = eg.Graph()
    
    # Add nodes
    alice = graph.add_node("Alice")
    bob = graph.add_node("Bob")
    charlie = graph.add_node("Charlie")
    
    # Add edges with data (e.g., relationship type)
    graph.add_edge(alice, bob, "friend")
    graph.add_edge(bob, charlie, "colleague")
    graph.add_edge(alice, charlie, "friend")
    
    print(f"Graph has {graph.node_count()} nodes and {graph.edge_count()} edges")

.. testoutput:: python

    Graph has 3 nodes and 3 edges

Accessing Graph Data
---------------------

Retrieving Node Data
^^^^^^^^^^^^^^^^^^^^

.. testcode:: python

    import egraph as eg

    graph = eg.Graph()
    alice = graph.add_node("Alice")
    bob = graph.add_node("Bob")
    
    # Get node data
    print(f"Node {alice}: {graph.node_weight(alice)}")
    print(f"Node {bob}: {graph.node_weight(bob)}")

.. testoutput:: python

    Node 0: Alice
    Node 1: Bob

Iterating Over Nodes
^^^^^^^^^^^^^^^^^^^^^

.. testcode:: python

    import egraph as eg

    graph = eg.Graph()
    graph.add_node("Alice")
    graph.add_node("Bob")
    graph.add_node("Charlie")
    
    # Iterate over all nodes
    for node_idx in graph.node_indices():
        data = graph.node_weight(node_idx)
        print(f"Node {node_idx}: {data}")

.. testoutput:: python

    Node 0: Alice
    Node 1: Bob
    Node 2: Charlie

Iterating Over Edges
^^^^^^^^^^^^^^^^^^^^^

.. testcode:: python

    import egraph as eg

    graph = eg.Graph()
    alice = graph.add_node("Alice")
    bob = graph.add_node("Bob")
    charlie = graph.add_node("Charlie")
    
    graph.add_edge(alice, bob, "friend")
    graph.add_edge(bob, charlie, "colleague")
    
    # Iterate over all edges
    for edge_idx in graph.edge_indices():
        source, target = graph.edge_endpoints(edge_idx)
        data = graph.edge_weight(edge_idx)
        print(f"Edge {source} -> {target}: {data}")

.. testoutput:: python

    Edge 0 -> 1: friend
    Edge 1 -> 2: colleague

Working with NetworkX
---------------------

egraph integrates seamlessly with NetworkX for graph creation and analysis.

Converting from NetworkX
^^^^^^^^^^^^^^^^^^^^^^^^^

.. testcode:: python

    import networkx as nx
    import egraph as eg

    # Create a NetworkX graph
    nx_graph = nx.karate_club_graph()
    
    # Convert to egraph
    graph = eg.Graph()
    node_map = {}
    
    for node in nx_graph.nodes:
        node_map[node] = graph.add_node(node)
    
    for u, v in nx_graph.edges:
        graph.add_edge(node_map[u], node_map[v], (u, v))
    
    print(f"Converted graph: {graph.node_count()} nodes, {graph.edge_count()} edges")

.. testoutput:: python

    Converted graph: 34 nodes, 78 edges

Using NetworkX Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use NetworkX for graph analysis and egraph for layout:

.. testcode:: python

    import networkx as nx
    import egraph as eg

    # Create and analyze with NetworkX
    nx_graph = nx.karate_club_graph()
    communities = nx.community.greedy_modularity_communities(nx_graph)
    
    # Convert to egraph for layout
    graph = eg.Graph()
    node_map = {}
    for node in nx_graph.nodes:
        node_map[node] = graph.add_node(node)
    for u, v in nx_graph.edges:
        graph.add_edge(node_map[u], node_map[v], (u, v))
    
    # Apply layout
    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
    sm = eg.StressMajorization(graph, drawing, lambda _: 100)
    sm.run(drawing)
    
    print(f"Found {len(communities)} communities")
    print(f"Layout computed for {graph.node_count()} nodes")

.. testoutput:: python

    Found 3 communities
    Layout computed for 34 nodes

Directed Graphs
---------------

DiGraph works similarly to Graph but maintains edge direction:

.. testcode:: python

    import egraph as eg

    digraph = eg.DiGraph()
    
    # Add nodes
    a = digraph.add_node("A")
    b = digraph.add_node("B")
    c = digraph.add_node("C")
    
    # Add directed edges
    digraph.add_edge(a, b, "depends_on")
    digraph.add_edge(b, c, "depends_on")
    digraph.add_edge(c, a, "depends_on")  # Creates a cycle
    
    print(f"DiGraph: {digraph.node_count()} nodes, {digraph.edge_count()} edges")

.. testoutput:: python

    DiGraph: 3 nodes, 3 edges

Best Practices
--------------

1. **Use meaningful node data**: Store relevant information in nodes for later reference
2. **Keep track of node indices**: Store the mapping between your data and node indices
3. **Choose the right graph type**: Use DiGraph only when direction matters
4. **Leverage NetworkX**: Use NetworkX for graph algorithms and egraph for layout

Next Steps
----------

* :doc:`02_layout_algorithms` - Learn about different layout algorithms
* :doc:`03_drawing_and_visualization` - Explore drawing spaces and visualization
* :doc:`../examples/index` - See more complex examples
