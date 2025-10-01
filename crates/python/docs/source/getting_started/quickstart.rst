Quick Start
===========

This guide will help you create your first graph visualization with egraph in just a few minutes.

Your First Graph Layout
-----------------------

Here's a minimal example that creates a simple graph and applies a layout algorithm:

.. testcode:: python

    import egraph as eg

    # Create a simple graph with 5 nodes
    graph = eg.Graph()
    
    # Add nodes
    n0 = graph.add_node(0)
    n1 = graph.add_node(1)
    n2 = graph.add_node(2)
    n3 = graph.add_node(3)
    n4 = graph.add_node(4)
    
    # Add edges to form a simple network
    graph.add_edge(n0, n1, (0, 1))
    graph.add_edge(n1, n2, (1, 2))
    graph.add_edge(n2, n3, (2, 3))
    graph.add_edge(n3, n4, (3, 4))
    graph.add_edge(n4, n0, (4, 0))
    
    # Create an initial 2D drawing with random positions
    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
    
    # Apply Stress Majorization layout algorithm
    sm = eg.StressMajorization(graph, drawing, lambda _: 100)
    sm.run(drawing)
    
    # Get the final positions
    positions = {i: (drawing.x(i), drawing.y(i)) for i in range(5)}
    # print("Node positions:", positions)

This example:

1. Creates a graph with 5 nodes
2. Connects them in a pentagon shape
3. Initializes random positions for the nodes
4. Applies the Stress Majorization algorithm to optimize the layout
5. Extracts the final node positions

Working with NetworkX
---------------------

egraph integrates seamlessly with NetworkX, a popular Python graph library:

.. testcode:: python

    import networkx as nx
    import egraph as eg
    import matplotlib.pyplot as plt

    # Create a graph using NetworkX
    nx_graph = nx.karate_club_graph()
    
    # Convert to egraph format
    graph = eg.Graph()
    indices = {}
    for node in nx_graph.nodes:
        indices[node] = graph.add_node(node)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))
    
    # Create initial drawing
    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
    
    # Apply layout algorithm
    sm = eg.StressMajorization(graph, drawing, lambda _: 100)
    sm.run(drawing)
    
    # Extract positions for NetworkX visualization
    pos = {node: (drawing.x(idx), drawing.y(idx)) 
           for node, idx in indices.items()}
    
    # Visualize with matplotlib
    nx.draw(nx_graph, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=10)
    # plt.savefig('karate_club.png')
    # plt.show()

Using Different Layout Algorithms
----------------------------------

egraph provides several layout algorithms. Here's how to use SGD (Stochastic Gradient Descent):

.. testcode:: python

    import egraph as eg

    # Create a graph (same as before)
    graph = eg.Graph()
    indices = [graph.add_node(i) for i in range(5)]
    graph.add_edge(indices[0], indices[1], (0, 1))
    graph.add_edge(indices[1], indices[2], (1, 2))
    graph.add_edge(indices[2], indices[3], (2, 3))
    graph.add_edge(indices[3], indices[4], (3, 4))
    graph.add_edge(indices[4], indices[0], (4, 0))
    
    # Create initial drawing
    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
    
    # Create random number generator for reproducibility
    rng = eg.Rng.seed_from(42)
    
    # Use SGD layout algorithm
    sgd = eg.FullSgd().build(graph, lambda _: 1.0)
    
    # Create a scheduler to control the optimization process
    scheduler = sgd.scheduler(100, 0.1)  # 100 iterations
    
    # Define the optimization step
    def step(eta):
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)
    
    # Run the optimization
    scheduler.run(step)
    
    # Get final positions
    positions = {i: (drawing.x(i), drawing.y(i)) for i in range(5)}
    # print("Final positions:", positions)

Key Concepts
------------

**Graph**: The data structure representing nodes and edges

**Drawing**: Stores the positions of nodes in a specific geometric space (Euclidean, Hyperbolic, Spherical, or Torus)

**Layout Algorithm**: Optimizes node positions to create an aesthetically pleasing visualization

**Scheduler**: Controls the optimization process for iterative algorithms like SGD

Next Steps
----------

Now that you've created your first graph layout, explore:

* :doc:`overview` - Learn more about egraph's features and capabilities
* :doc:`../tutorial/01_graph_basics` - Deep dive into graph creation and manipulation
* :doc:`../examples/index` - See more examples of different layout algorithms
