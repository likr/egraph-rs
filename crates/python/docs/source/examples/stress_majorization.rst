Stress Majorization
===================

This example demonstrates how to use the Stress Majorization layout algorithm.

Basic Stress Majorization Example
------------------------------------------

.. testcode:: python

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
    
    # Create a StressMajorization instance
    sm = eg.StressMajorization(graph, drawing, lambda _: 100)
    
    # Set convergence parameters
    sm.epsilon = 1e-4  # Convergence threshold
    sm.max_iterations = 200  # Maximum number of iterations
    
    # Run the algorithm
    sm.run(drawing)

    # Extract node positions
    pos = {u: (drawing.x(i), drawing.y(i)) for u, i in indices.items()}
    
    # Visualize with NetworkX
    nx.draw(nx_graph, pos)

Using a Distance Matrix
---------------------------

For more control, you can create a StressMajorization instance from a distance matrix:

.. testcode:: python

    # Create a distance matrix
    distance_matrix = eg.DistanceMatrix(graph)
    
    # Optionally, modify distances
    for i in range(graph.node_count()):
        for j in range(i + 1, graph.node_count()):
            # Set custom distances
            distance = distance_matrix.get(i, j)
            # Modify the distance if needed
            distance_matrix.set(i, j, distance)
            distance_matrix.set(j, i, distance)  # For undirected graphs
    
    # Create a StressMajorization instance from the distance matrix
    sm = eg.StressMajorization.with_distance_matrix(drawing, distance_matrix)
    
    # Run the algorithm
    sm.run(drawing)

Applying a Single Iteration
----------------------------------

You can also apply a single iteration of the algorithm and check the stress value:

.. testcode:: python

    # Create a fresh drawing for this example
    drawing_fresh = eg.DrawingEuclidean2d.initial_placement(graph)
    sm_fresh = eg.StressMajorization(graph, drawing_fresh, lambda _: 100)
    
    # Apply a single iteration
    stress = sm_fresh.apply(drawing_fresh)
    
    # Apply multiple iterations manually
    for i in range(10):
        stress = sm_fresh.apply(drawing_fresh)
