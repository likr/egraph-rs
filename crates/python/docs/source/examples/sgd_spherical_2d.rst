Spherical 2D Stochastic Gradient Descent
========================================

This example demonstrates how to use the Stochastic Gradient Descent (SGD) layout algorithm with spherical 2D drawings.

Basic Spherical SGD Example
----------------------------------

.. code-block:: python

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

    # Create a spherical drawing
    drawing = eg.DrawingSpherical2d(graph)
    
    # Initialize with random positions on the sphere
    for i in range(graph.node_count()):
        # Generate random positions in spherical coordinates
        # Longitude (0 to 2π)
        lon = 2 * np.pi * np.random.random() - np.pi
        # Latitude (-π/2 to π/2)
        lat = np.pi * np.random.random() - np.pi/2
        
        # Set the coordinates
        drawing.longitude(i, lon)
        drawing.latitude(i, lat)
    
    # Create a random number generator with a seed for reproducibility
    rng = eg.Rng.seed_from(0)
    
    # Create a SparseSgd instance
    sgd = eg.SparseSgd(
        graph,
        lambda _: 0.5,  # edge length (in radians)
        50,  # number of pivots
        rng,
    )
    
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
        lon = drawing.longitude(i)
        lat = drawing.latitude(i)
        
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
    
    plt.savefig('sgd_spherical_2d_layout.png')
    plt.show()

Working with Spherical Distances
----------------------------------

When working with spherical space, distances are measured along great circles:

.. code-block:: python

    # Calculate great-circle distance between two points on a sphere
    def spherical_distance(lon1, lat1, lon2, lat2):
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return c
    
    # Example usage
    node1 = 0
    node2 = 1
    lon1, lat1 = drawing.longitude(node1), drawing.latitude(node1)
    lon2, lat2 = drawing.longitude(node2), drawing.latitude(node2)
    
    dist = spherical_distance(lon1, lat1, lon2, lat2)
    print(f"Spherical distance between nodes {node1} and {node2}: {dist} radians")
