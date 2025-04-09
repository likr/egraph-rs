Overwrap Removal
==============

This example demonstrates how to use the Overwrap Removal algorithm to resolve node overlaps in a graph layout.

Basic Overwrap Removal Example
---------------------------

.. code-block:: python

    import networkx as nx
    import egraph as eg
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a graph from NetworkX
    nx_graph = nx.les_miserables_graph()
    graph = eg.Graph()
    indices = {}
    for u in nx_graph.nodes:
        indices[u] = graph.add_node(u)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))

    # Create an initial drawing
    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
    
    # Apply a layout algorithm (e.g., StressMajorization)
    sm = eg.StressMajorization(graph, drawing, lambda _: 100)
    sm.run(drawing)
    
    # Save the positions before overlap removal
    pos_before = {u: (drawing.x(i), drawing.y(i)) for u, i in indices.items()}
    
    # Create node sizes (radius for each node)
    node_sizes = np.ones(graph.node_count()) * 0.1
    
    # Create an OverwrapRemoval instance
    # The first parameter is the node radius function
    or_algo = eg.OverwrapRemoval(lambda i: node_sizes[i])
    
    # Run the algorithm
    or_algo.run(drawing)
    
    # Save the positions after overlap removal
    pos_after = {u: (drawing.x(i), drawing.y(i)) for u, i in indices.items()}
    
    # Visualize the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Draw the graph before overlap removal
    nx.draw(nx_graph, pos_before, ax=ax1, node_size=200)
    ax1.set_title('Before Overlap Removal')
    
    # Draw the graph after overlap removal
    nx.draw(nx_graph, pos_after, ax=ax2, node_size=200)
    ax2.set_title('After Overlap Removal')
    
    plt.savefig('overwrap_removal.png')
    plt.show()

Using Variable Node Sizes
----------------------

You can also use variable node sizes:

.. code-block:: python

    # Create variable node sizes based on node degree
    node_sizes = np.zeros(graph.node_count())
    for u, i in indices.items():
        # Set node size proportional to degree
        node_sizes[i] = 0.05 + 0.02 * nx_graph.degree(u)
    
    # Create an OverwrapRemoval instance with variable node sizes
    or_algo = eg.OverwrapRemoval(lambda i: node_sizes[i])
    
    # Run the algorithm
    or_algo.run(drawing)

Controlling the Overlap Removal Process
------------------------------------

You can control the overlap removal process by setting parameters:

.. code-block:: python

    # Create an OverwrapRemoval instance with custom parameters
    or_algo = eg.OverwrapRemoval(
        lambda i: node_sizes[i],  # Node radius function
    )
    
    # Apply a single iteration
    or_algo.apply(drawing)
    
    # Apply multiple iterations manually
    for i in range(10):
        or_algo.apply(drawing)
        # You can check the layout after each iteration
        # and stop when satisfied
