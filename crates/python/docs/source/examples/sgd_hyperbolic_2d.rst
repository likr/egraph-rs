Hyperbolic 2D Stochastic Gradient Descent
=========================================

This example demonstrates how to use the Stochastic Gradient Descent (SGD) layout algorithm with hyperbolic 2D drawings.

Basic Hyperbolic SGD Example
----------------------------------

.. testcode:: python

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

    # Create a hyperbolic drawing using the factory method
    drawing = eg.DrawingHyperbolic2d.initial_placement(graph)
    
    # Create a random number generator with a seed for reproducibility
    rng = eg.Rng.seed_from(0)
    
    # Create a SparseSgd instance using the builder pattern
    sgd = eg.SparseSgd().h(50).build(graph, lambda _: 0.3, rng)
    
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

Working with Hyperbolic Distances
----------------------------------

When working with hyperbolic space, it's important to understand that distances are different from Euclidean space.
The hyperbolic distance formula in the Poincaré disk model can be calculated using the positions of nodes.
