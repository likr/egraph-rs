Stochastic Gradient Descent (SGD)
==================================

This example demonstrates how to use the Stochastic Gradient Descent (SGD) layout algorithm.

Basic SGD Example
-----------------------

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
    
    # Create a random number generator with a seed for reproducibility
    rng = eg.Rng.seed_from(0)
    
    # Create a SparseSgd instance using the builder pattern
    sgd = eg.SparseSgd().h(50).build(graph, lambda _: 30, rng)
    
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
    
    # Visualize with NetworkX
    nx.draw(nx_graph, pos)

Using FullSgd
-------------------

For smaller graphs, you can use `FullSgd` which computes all-pairs shortest path distances:

.. testcode:: python

    # Create a FullSgd instance using the builder pattern
    sgd = eg.FullSgd().build(graph, lambda _: 30)
    
    # The rest of the code is the same as the SparseSgd example
    scheduler = sgd.scheduler(100, 0.1)
    
    def step(eta):
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)
    
    scheduler.run(step)
