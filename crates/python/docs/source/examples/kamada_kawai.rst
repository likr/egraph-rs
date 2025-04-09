Kamada-Kawai
===========

This example demonstrates how to use the Kamada-Kawai layout algorithm.

Basic Kamada-Kawai Example
-----------------------

.. code-block:: python

    import networkx as nx
    import egraph as eg
    import matplotlib.pyplot as plt

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
    
    # Create a KamadaKawai instance
    # The lambda function defines the desired distance between nodes
    # Here we use a constant distance of 1.0 for all edges
    kk = eg.KamadaKawai(graph, lambda _: 1.0)
    
    # Set the convergence threshold
    kk.epsilon = 1e-3
    
    # Run the algorithm
    kk.run(drawing)

    # Extract node positions
    pos = {u: (drawing.x(i), drawing.y(i)) for u, i in indices.items()}
    
    # Visualize with NetworkX
    nx.draw(nx_graph, pos)
    plt.savefig('kamada_kawai_layout.png')
    plt.show()

Using Custom Edge Distances
------------------------

You can customize the desired distances between nodes:

.. code-block:: python

    # Create a KamadaKawai instance with custom edge distances
    # The lambda function takes an edge index and returns the desired distance
    kk = eg.KamadaKawai(graph, lambda e: 2.0 if graph.edge_weight(e) > 5 else 1.0)
    
    # Run the algorithm
    kk.run(drawing)

Applying to a Single Node
----------------------

You can also apply the algorithm to a single node:

.. code-block:: python

    # Apply the algorithm to a specific node
    node_index = 0
    kk.apply_node(drawing, node_index)
    
    # Apply the algorithm to all nodes one by one
    for i in range(graph.node_count()):
        kk.apply_node(drawing, i)
