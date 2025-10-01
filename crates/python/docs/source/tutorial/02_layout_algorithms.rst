Layout Algorithms
=================

This tutorial explores the different layout algorithms available in egraph and when to use each one.

Understanding Layout Algorithms
--------------------------------

Layout algorithms compute positions for graph nodes to create aesthetically pleasing visualizations. Different algorithms have different strengths:

* **Speed vs Quality**: Some algorithms are faster but produce lower quality layouts
* **Graph Size**: Some algorithms scale better to large graphs
* **Graph Structure**: Some algorithms work better for specific graph types

Stress Majorization
-------------------

Stress Majorization is an iterative algorithm that minimizes the stress function, producing high-quality layouts.

**Best for**: Small to medium graphs where quality is important

.. testcode:: python

    import egraph as eg
    import networkx as nx

    # Create a graph
    nx_graph = nx.karate_club_graph()
    graph = eg.Graph()
    indices = {}
    for node in nx_graph.nodes:
        indices[node] = graph.add_node(node)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))
    
    # Create initial drawing
    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
    
    # Apply Stress Majorization
    sm = eg.StressMajorization(graph, drawing, lambda _: 100)
    sm.run(drawing)
    
    print(f"Layout computed for {graph.node_count()} nodes")

.. testoutput:: python

    Layout computed for 34 nodes

**Parameters**:
- `graph`: The graph to layout
- `drawing`: Initial node positions
- `length`: Function that returns desired edge length (lambda edge_index: length)

Kamada-Kawai
------------

Kamada-Kawai is a spring-based algorithm that treats edges as springs.

**Best for**: Small graphs, tree-like structures

.. testcode:: python

    import egraph as eg

    # Create a simple graph
    graph = eg.Graph()
    nodes = [graph.add_node(i) for i in range(6)]
    graph.add_edge(nodes[0], nodes[1], (0, 1))
    graph.add_edge(nodes[0], nodes[2], (0, 2))
    graph.add_edge(nodes[1], nodes[3], (1, 3))
    graph.add_edge(nodes[1], nodes[4], (1, 4))
    graph.add_edge(nodes[2], nodes[5], (2, 5))
    
    # Apply Kamada-Kawai
    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
    kk = eg.KamadaKawai(graph, lambda _: 100)
    kk.run(drawing)
    

.. testoutput:: python


SGD (Stochastic Gradient Descent)
----------------------------------

SGD algorithms are fast and scalable, making them ideal for large graphs.

FullSgd
^^^^^^^

Computes all-pairs shortest paths for accurate layouts.

**Best for**: Small to medium graphs where accuracy is important

.. testcode:: python

    import egraph as eg

    graph = eg.Graph()
    nodes = [graph.add_node(i) for i in range(10)]
    for i in range(9):
        graph.add_edge(nodes[i], nodes[i+1], (i, i+1))
    
    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
    rng = eg.Rng.seed_from(42)
    
    # Create FullSgd instance
    sgd = eg.FullSgd().build(graph, lambda _: 30)
    
    # Run with scheduler
    scheduler = sgd.scheduler(100, 0.1)
    def step(eta):
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)
    scheduler.run(step)
    
    print("FullSgd layout complete")

.. testoutput:: python

    FullSgd layout complete

SparseSgd
^^^^^^^^^

Uses pivot nodes for efficient computation on large graphs.

**Best for**: Large graphs (hundreds to thousands of nodes)

.. testcode:: python

    import egraph as eg
    import networkx as nx

    # Create a larger graph
    nx_graph = nx.les_miserables_graph()
    graph = eg.Graph()
    indices = {}
    for node in nx_graph.nodes:
        indices[node] = graph.add_node(node)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))
    
    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
    rng = eg.Rng.seed_from(42)
    
    # Create SparseSgd with 50 pivot nodes
    sgd = eg.SparseSgd().h(50).build(graph, lambda _: 30, rng)
    
    # Run optimization
    scheduler = sgd.scheduler(100, 0.1)
    def step(eta):
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)
    scheduler.run(step)
    
    print(f"SparseSgd layout for {graph.node_count()} nodes complete")

.. testoutput:: python

    SparseSgd layout for 77 nodes complete

**Parameters**:
- `h(n)`: Number of pivot nodes (default: 30)

Omega
^^^^^

Uses spectral coordinates for initialization, combining spectral methods with SGD.

**Best for**: Graphs with clear community structure

.. testcode:: python

    import egraph as eg
    import networkx as nx

    nx_graph = nx.karate_club_graph()
    graph = eg.Graph()
    for node in nx_graph.nodes:
        graph.add_node(node)
    for u, v in nx_graph.edges:
        graph.add_edge(u, v, (u, v))
    
    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
    rng = eg.Rng.seed_from(42)
    
    # Create Omega instance
    omega = eg.Omega().d(2).k(30).build(graph, lambda _: 30, rng)
    
    # Run optimization
    scheduler = omega.scheduler(100, 0.1)
    def step(eta):
        omega.shuffle(rng)
        omega.apply(drawing, eta)
    scheduler.run(step)
    
    print("Omega layout complete")

.. testoutput:: python

    Omega layout complete

**Parameters**:
- `d(n)`: Number of spectral dimensions (default: 2)
- `k(n)`: Number of random pairs per node (default: 30)

MDS (Multidimensional Scaling)
-------------------------------

MDS algorithms preserve distances in lower dimensions.

ClassicalMds
^^^^^^^^^^^^

Uses eigendecomposition for exact distance preservation.

**Best for**: Small graphs where distance preservation is critical

.. testcode:: python

    import egraph as eg

    graph = eg.Graph()
    nodes = [graph.add_node(i) for i in range(8)]
    for i in range(7):
        graph.add_edge(nodes[i], nodes[i+1], (i, i+1))
    
    # Apply Classical MDS
    mds = eg.ClassicalMds(graph, lambda _: 1.0)
    drawing = mds.run(2)
    

.. testoutput:: python


PivotMds
^^^^^^^^

Scalable variant using landmark nodes.

**Best for**: Large graphs where approximate distance preservation is acceptable

.. testcode:: python

    import egraph as eg
    import networkx as nx

    nx_graph = nx.les_miserables_graph()
    graph = eg.Graph()
    indices = {}
    for node in nx_graph.nodes:
        indices[node] = graph.add_node(node)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))
    
    # Use 20 pivot nodes with PivotMds
    # Create a list of pivot node indices
    import random
    random.seed(42)
    pivot_indices = random.sample(range(graph.node_count()), min(20, graph.node_count()))
    
    mds = eg.PivotMds(graph, lambda _: 1.0, pivot_indices)
    drawing = mds.run(2)

Choosing the Right Algorithm
-----------------------------

Use this decision tree to select an algorithm:

1. **Graph size < 100 nodes**
   
   * High quality needed → Stress Majorization
   * Tree-like structure → Kamada-Kawai
   * Fast computation → FullSgd

2. **Graph size 100-1000 nodes**
   
   * Community structure → Omega
   * General purpose → SparseSgd
   * Distance preservation → PivotMds

3. **Graph size > 1000 nodes**
   
   * Use SparseSgd or PivotMds
   * Increase pivot nodes for better quality

Customizing Edge Lengths
-------------------------

All algorithms accept a length function to customize edge lengths:

.. testcode:: python

    import egraph as eg

    graph = eg.Graph()
    nodes = [graph.add_node(i) for i in range(5)]
    
    # Add edges with different weights
    e1 = graph.add_edge(nodes[0], nodes[1], 1.0)
    e2 = graph.add_edge(nodes[1], nodes[2], 2.0)
    e3 = graph.add_edge(nodes[2], nodes[3], 1.0)
    e4 = graph.add_edge(nodes[3], nodes[4], 3.0)
    
    # Pre-fetch edge weights to avoid borrow conflicts
    edge_weights = {i: graph.edge_weight(i) for i in range(graph.edge_count())}
    
    # Use edge weights as lengths
    def edge_length(edge_idx):
        return edge_weights[edge_idx]
    
    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
    sm = eg.StressMajorization(graph, drawing, edge_length)
    sm.run(drawing)
    

.. testoutput:: python


Next Steps
----------

* :doc:`03_drawing_and_visualization` - Learn about drawing spaces and visualization
* :doc:`../examples/index` - See algorithm-specific examples
* :doc:`../api/layout` - Detailed API reference
