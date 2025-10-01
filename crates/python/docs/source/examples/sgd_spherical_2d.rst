Spherical 2D Stochastic Gradient Descent
========================================

This example demonstrates how to use the Stochastic Gradient Descent (SGD) layout algorithm with spherical 2D drawings.

Basic Spherical SGD Example
----------------------------------

.. testcode:: python

    import networkx as nx
    import egraph as eg
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D

    # Create a graph from NetworkX
    nx_graph = nx.les_miserables_graph()
    graph = eg.Graph()
    indices = {}
    for u in nx_graph.nodes:
        indices[u] = graph.add_node(u)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))

    # Create a spherical drawing using the factory method
    drawing = eg.DrawingSpherical2d.initial_placement(graph)
    
    # Create a random number generator with a seed for reproducibility
    rng = eg.Rng.seed_from(0)
    
    # Create a SparseSgd instance using the builder pattern
    sgd = eg.SparseSgd().h(50).build(graph, lambda _: 0.5, rng)
    
    # Create a scheduler for the SGD algorithm
    scheduler = sgd.scheduler(
        100,  # number of iterations
        0.1,  # eps: eta_min = eps * min d[i, j] ^ 2
    )

    # Define a step function for the scheduler
    def step(eta):
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)
    
    # Run the scheduler
    scheduler.run(step)

    # Extract node positions in 3D Cartesian coordinates for visualization
    pos_3d = {}
    for u, i in indices.items():
        lon = drawing.lon(i)
        lat = drawing.lat(i)
        
        # Convert spherical to Cartesian coordinates
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        
        pos_3d[u] = (x, y, z)
    
    # Visualize with Matplotlib's 3D plotting
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw the sphere wireframe
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.2)
    
    # Plot nodes
    for node, (x, y, z) in pos_3d.items():
        ax.scatter(x, y, z, c='b', s=30)
    
    # Plot edges
    for u, v in nx_graph.edges():
        x = [pos_3d[u][0], pos_3d[v][0]]
        y = [pos_3d[u][1], pos_3d[v][1]]
        z = [pos_3d[u][2], pos_3d[v][2]]
        ax.plot(x, y, z, c='k', alpha=0.5)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])

Working with Spherical Distances
----------------------------------

When working with spherical space, distances are measured along great circles.
The great-circle distance (also known as orthodromic distance) is the shortest distance
between two points on the surface of a sphere, measured along the surface.
