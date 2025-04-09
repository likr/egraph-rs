Examples
========

This section provides examples of how to use the egraph library.

.. toctree::
   :maxdepth: 2

   sgd
   stress_majorization
   kamada_kawai
   sgd_3d
   sgd_hyperbolic_2d
   sgd_spherical_2d
   sgd_torus
   overwrap_removal

Basic Usage
-----------------

Here's a simple example of creating a graph and applying a layout algorithm:

.. code-block:: python

    import networkx as nx
    import egraph as eg
    
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
    
    # Apply a layout algorithm
    sm = eg.StressMajorization(graph, drawing, lambda _: 100)
    sm.run(drawing)
    
    # Extract node positions
    pos = {u: (drawing.x(i), drawing.y(i)) for u, i in indices.items()}
    
    # Visualize with NetworkX
    import matplotlib.pyplot as plt
    nx.draw(nx_graph, pos)
    plt.show()
