Torus 2D Stochastic Gradient Descent
=====================================

This example demonstrates how to use the Stochastic Gradient Descent (SGD) layout algorithm with torus 2D drawings.

Basic Torus SGD Example
---------------------------

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

    # Create a torus drawing using the factory method
    drawing = eg.DrawingTorus2d.initial_placement(graph)
    
    # Create a random number generator with a seed for reproducibility
    rng = eg.Rng.seed_from(0)
    
    # Create a SparseSgd instance using the builder pattern
    sgd = eg.SparseSgd().h(50).build(graph, lambda _: 0.2, rng)
    
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
            for ((sx1, sy1), (sx2, sy2)) in segments:
                ax.plot([sx1, sx2], [sy1, sy2], 'k-', alpha=0.5)
    
    # Set limits and aspect ratio
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')

Visualizing the Torus in 3D
----------------------------------

The torus topology can be visualized in 3D by mapping the 2D coordinates to a 3D torus surface.
The torus has two radii: the major radius (distance from the center of the tube to the center
of the torus) and the minor radius (radius of the tube itself).
