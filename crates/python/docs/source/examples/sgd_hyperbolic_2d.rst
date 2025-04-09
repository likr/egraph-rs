Hyperbolic 2D Stochastic Gradient Descent
=========================================

This example demonstrates how to use the Stochastic Gradient Descent (SGD) layout algorithm with hyperbolic 2D drawings.

Basic Hyperbolic SGD Example
----------------------------------

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

    # Create a hyperbolic drawing
    drawing = eg.DrawingHyperbolic2d(graph)
    
    # Initialize with random positions within the Poincaré disk
    for i in range(graph.node_count()):
        # Generate random positions in polar coordinates
        r = 0.9 * np.random.random()  # Radius (keep within unit disk)
        theta = 2 * np.pi * np.random.random()  # Angle
        
        # Convert to Cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Set the coordinates
        drawing.x(i, x)
        drawing.y(i, y)
    
    # Create a random number generator with a seed for reproducibility
    rng = eg.Rng.seed_from(0)
    
    # Create a SparseSgd instance
    sgd = eg.SparseSgd(
        graph,
        lambda _: 0.3,  # edge length (smaller in hyperbolic space)
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
    
    # Draw the Poincaré disk boundary
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    ax.add_patch(circle)
    
    # Draw the graph
    nx.draw(nx_graph, pos, ax=ax, node_size=50)
    
    # Set equal aspect ratio and limits
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    
    plt.savefig('sgd_hyperbolic_2d_layout.png')
    plt.show()

Working with Hyperbolic Distances
----------------------------------

When working with hyperbolic space, it's important to understand that distances are different from Euclidean space:

.. code-block:: python

    # Calculate hyperbolic distance between two points in the Poincaré disk
    def hyperbolic_distance(x1, y1, x2, y2):
        # Convert to complex numbers for easier calculation
        z1 = complex(x1, y1)
        z2 = complex(x2, y2)
        
        # Calculate the Möbius addition
        numerator = abs(z1 - z2)
        denominator = (1 - abs(z1)**2) * (1 - abs(z2)**2)
        
        # Return the hyperbolic distance
        return 2 * np.arctanh(numerator / denominator)
    
    # Example usage
    node1 = 0
    node2 = 1
    x1, y1 = drawing.x(node1), drawing.y(node1)
    x2, y2 = drawing.x(node2), drawing.y(node2)
    
    dist = hyperbolic_distance(x1, y1, x2, y2)
    print(f"Hyperbolic distance between nodes {node1} and {node2}: {dist}")
