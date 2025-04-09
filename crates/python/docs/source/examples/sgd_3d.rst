3D Stochastic Gradient Descent
===========================

This example demonstrates how to use the Stochastic Gradient Descent (SGD) layout algorithm with 3D Euclidean drawings.

Basic 3D SGD Example
-----------------

.. code-block:: python

    import networkx as nx
    import egraph as eg
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Create a graph from NetworkX
    nx_graph = nx.les_miserables_graph()
    graph = eg.Graph()
    indices = {}
    for u in nx_graph.nodes:
        indices[u] = graph.add_node(u)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))

    # Create a 3D drawing
    # The second parameter (3) specifies the number of dimensions
    drawing = eg.DrawingEuclidean(graph, 3)
    
    # Initialize with random positions
    for i in range(graph.node_count()):
        drawing.set(i, 0, 2.0 * (0.5 - i / graph.node_count()))
        drawing.set(i, 1, 2.0 * (0.5 - (i % 10) / 10))
        drawing.set(i, 2, 2.0 * (0.5 - (i % 20) / 20))
    
    # Create a random number generator with a seed for reproducibility
    rng = eg.Rng.seed_from(0)
    
    # Create a SparseSgd instance
    sgd = eg.SparseSgd(
        graph,
        lambda _: 30,  # edge length
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
    pos_3d = {u: (drawing.get(i, 0), drawing.get(i, 1), drawing.get(i, 2)) 
              for u, i in indices.items()}
    
    # Visualize with Matplotlib's 3D plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot nodes
    for node, (x, y, z) in pos_3d.items():
        ax.scatter(x, y, z, c='b', s=30)
    
    # Plot edges
    for u, v in nx_graph.edges():
        x = [pos_3d[u][0], pos_3d[v][0]]
        y = [pos_3d[u][1], pos_3d[v][1]]
        z = [pos_3d[u][2], pos_3d[v][2]]
        ax.plot(x, y, z, c='k', alpha=0.5)
    
    plt.savefig('sgd_3d_layout.png')
    plt.show()

Using ClassicalMds for 3D Initialization
-------------------------------------

You can also use ClassicalMds to create an initial 3D layout:

.. code-block:: python

    # Create a 3D drawing
    drawing = eg.DrawingEuclidean(graph, 3)
    
    # Use ClassicalMds to create an initial layout
    mds = eg.ClassicalMds(graph, lambda _: 1.0)
    mds.run(drawing)
    
    # Then apply SGD to refine the layout
    sgd = eg.SparseSgd(graph, lambda _: 30, 50, rng)
    scheduler = sgd.scheduler(100, 0.1)
    scheduler.run(lambda eta: (sgd.shuffle(rng), sgd.apply(drawing, eta)))
