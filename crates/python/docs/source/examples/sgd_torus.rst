Torus 2D Stochastic Gradient Descent
=====================================

This example demonstrates how to use the Stochastic Gradient Descent (SGD) layout algorithm with torus 2D drawings.

Basic Torus SGD Example
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

    # Create a torus drawing
    drawing = eg.DrawingTorus2d(graph)
    
    # Initialize with random positions on the torus
    for i in range(graph.node_count()):
        # Generate random positions in [0, 1) range
        x = np.random.random()
        y = np.random.random()
        
        # Set the coordinates
        drawing.x(i, x)
        drawing.y(i, y)
    
    # Create a random number generator with a seed for reproducibility
    rng = eg.Rng.seed_from(0)
    
    # Create a SparseSgd instance
    sgd = eg.SparseSgd(
        graph,
        lambda _: 0.2,  # edge length
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

    # Extract node positions
    pos = {u: (drawing.x(i), drawing.y(i)) for u, i in indices.items()}
    
    # Visualize with NetworkX and Matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw the torus boundary
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, color='gray', linestyle='--'))
    
    # Draw the graph
    nx.draw(nx_graph, pos, ax=ax, node_size=50)
    
    # Draw edges that cross the boundary
    for u, v in nx_graph.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        
        # Check if the edge crosses the boundary
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        if dx > 0.5 or dy > 0.5:
            # This edge crosses the boundary, draw it as a pair of segments
            segments = drawing.edge_segments(indices[u], indices[v])
            for (sx1, sy1, sx2, sy2) in segments:
                ax.plot([sx1, sx2], [sy1, sy2], 'k-', alpha=0.5)
    
    # Set limits and aspect ratio
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    
    plt.savefig('sgd_torus_layout.png')
    plt.show()

Visualizing the Torus in 3D
----------------------------------

You can also visualize the torus layout in 3D to better understand the topology:

.. code-block:: python

    # Convert 2D torus coordinates to 3D torus coordinates
    def torus_to_3d(x, y, R=2, r=1):
        """
        Convert 2D torus coordinates to 3D coordinates.
        R: major radius (distance from center of tube to center of torus)
        r: minor radius (radius of the tube)
        """
        theta = 2 * np.pi * x  # angle around the center of the torus
        phi = 2 * np.pi * y    # angle around the tube
        
        X = (R + r * np.cos(phi)) * np.cos(theta)
        Y = (R + r * np.cos(phi)) * np.sin(theta)
        Z = r * np.sin(phi)
        
        return X, Y, Z
    
    # Create 3D positions
    pos_3d = {}
    for u, i in indices.items():
        x, y = drawing.x(i), drawing.y(i)
        pos_3d[u] = torus_to_3d(x, y)
    
    # Visualize in 3D
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw the torus wireframe
    u, v = np.mgrid[0:2*np.pi:20j, 0:2*np.pi:20j]
    R, r = 2, 1  # Major and minor radii
    X = (R + r * np.cos(v)) * np.cos(u)
    Y = (R + r * np.cos(v)) * np.sin(u)
    Z = r * np.sin(v)
    ax.plot_wireframe(X, Y, Z, color="gray", alpha=0.2)
    
    # Plot nodes
    for node, (x, y, z) in pos_3d.items():
        ax.scatter(x, y, z, c='b', s=30)
    
    # Plot edges
    for u, v in nx_graph.edges():
        x = [pos_3d[u][0], pos_3d[v][0]]
        y = [pos_3d[u][1], pos_3d[v][1]]
        z = [pos_3d[u][2], pos_3d[v][2]]
        ax.plot(x, y, z, c='k', alpha=0.5)
    
    plt.savefig('sgd_torus_3d_layout.png')
    plt.show()
