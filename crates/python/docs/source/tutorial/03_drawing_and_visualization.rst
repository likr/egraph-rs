Drawing and Visualization
=========================

This tutorial covers drawing spaces, visualization techniques, and integration with matplotlib.

Understanding Drawing Spaces
-----------------------------

egraph supports multiple geometric spaces for graph layout:

* **Euclidean**: Standard 2D and n-dimensional space
* **Hyperbolic**: Poincaré disk model for hierarchical graphs
* **Spherical**: Sphere surface for global connectivity
* **Torus**: Periodic boundaries for wrapping layouts

Euclidean 2D Drawing
--------------------

The most common drawing space for standard graph visualization.

Creating a Drawing
^^^^^^^^^^^^^^^^^^

.. testcode:: python

    import egraph as eg

    graph = eg.Graph()
    nodes = [graph.add_node(i) for i in range(5)]
    for i in range(4):
        graph.add_edge(nodes[i], nodes[i+1], (i, i+1))
    
    # Create 2D Euclidean drawing with random initial positions
    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
    
    print(f"Created 2D drawing for {graph.node_count()} nodes")

.. testoutput:: python

    Created 2D drawing for 5 nodes

Accessing Coordinates
^^^^^^^^^^^^^^^^^^^^^

.. testcode:: python

    import egraph as eg

    graph = eg.Graph()
    nodes = [graph.add_node(i) for i in range(3)]
    
    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
    
    # Get coordinates for each node
    for i in range(3):
        x = drawing.x(i)
        y = drawing.y(i)
        print(f"Node {i}: ({x:.2f}, {y:.2f})")

.. testoutput:: python
    :options: +SKIP

    Node 0: (0.00, 0.00)
    Node 1: (1.23, 4.56)
    Node 2: (-2.34, 3.45)

Setting Coordinates
^^^^^^^^^^^^^^^^^^^

.. testcode:: python

    import egraph as eg

    graph = eg.Graph()
    nodes = [graph.add_node(i) for i in range(3)]
    
    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
    
    # Set specific positions
    drawing.set_x(0, 0.0)
    drawing.set_y(0, 0.0)
    drawing.set_x(1, 1.0)
    drawing.set_y(1, 0.0)
    drawing.set_x(2, 0.5)
    drawing.set_y(2, 1.0)
    
    print("Coordinates set manually")

.. testoutput:: python

    Coordinates set manually

N-Dimensional Euclidean Drawing
--------------------------------

For higher-dimensional layouts (3D, 4D, etc.), see the :doc:`../examples/sgd_3d` example which demonstrates 3D graph layouts using SGD algorithms.

Hyperbolic 2D Drawing
---------------------

Hyperbolic space is ideal for hierarchical graphs and trees.

.. testcode:: python

    import egraph as eg

    # Create a tree-like graph
    graph = eg.Graph()
    root = graph.add_node(0)
    
    # Add two levels
    level1 = [graph.add_node(i) for i in range(1, 4)]
    for node in level1:
        graph.add_edge(root, node, (root, node))
    
    level2 = [graph.add_node(i) for i in range(4, 10)]
    for i, node in enumerate(level2):
        parent = level1[i % 3]
        graph.add_edge(parent, node, (parent, node))
    
    # Create hyperbolic drawing
    drawing = eg.DrawingHyperbolic2d.initial_placement(graph)
    
    # Apply layout
    rng = eg.Rng.seed_from(42)
    sgd = eg.SparseSgd().h(5).build(graph, lambda _: 0.5, rng)
    scheduler = sgd.scheduler(50, 0.1)
    def step(eta):
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)
    scheduler.run(step)
    
    print(f"Hyperbolic layout for {graph.node_count()} nodes")

.. testoutput:: python

    Hyperbolic layout for 10 nodes

Spherical 2D Drawing
--------------------

Spherical layouts are useful for global networks.

.. testcode:: python

    import egraph as eg
    import networkx as nx

    # Create a graph
    nx_graph = nx.karate_club_graph()
    graph = eg.Graph()
    for node in nx_graph.nodes:
        graph.add_node(node)
    for u, v in nx_graph.edges:
        graph.add_edge(u, v, (u, v))
    
    # Create spherical drawing
    drawing = eg.DrawingSpherical2d.initial_placement(graph)
    
    # Apply layout
    rng = eg.Rng.seed_from(42)
    sgd = eg.SparseSgd().h(10).build(graph, lambda _: 0.3, rng)
    scheduler = sgd.scheduler(50, 0.1)
    def step(eta):
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)
    scheduler.run(step)
    
    print(f"Spherical layout for {graph.node_count()} nodes")

.. testoutput:: python

    Spherical layout for 34 nodes

Torus 2D Drawing
----------------

Torus layouts have periodic boundaries, useful for certain network types.

.. testcode:: python

    import egraph as eg

    graph = eg.Graph()
    nodes = [graph.add_node(i) for i in range(10)]
    for i in range(9):
        graph.add_edge(nodes[i], nodes[i+1], (i, i+1))
    graph.add_edge(nodes[9], nodes[0], (9, 0))  # Wrap around
    
    # Create torus drawing
    drawing = eg.DrawingTorus2d.initial_placement(graph)
    
    # Apply layout
    rng = eg.Rng.seed_from(42)
    sgd = eg.SparseSgd().h(5).build(graph, lambda _: 0.1, rng)
    scheduler = sgd.scheduler(50, 0.1)
    def step(eta):
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)
    scheduler.run(step)
    
    print(f"Torus layout for {graph.node_count()} nodes")

.. testoutput:: python

    Torus layout for 10 nodes

Visualization with Matplotlib
------------------------------

Basic Visualization
^^^^^^^^^^^^^^^^^^^

.. testcode:: python

    import networkx as nx
    import egraph as eg
    import matplotlib.pyplot as plt

    # Create and layout graph
    nx_graph = nx.karate_club_graph()
    graph = eg.Graph()
    indices = {}
    for node in nx_graph.nodes:
        indices[node] = graph.add_node(node)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))
    
    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
    sm = eg.StressMajorization(graph, drawing, lambda _: 100)
    sm.run(drawing)
    
    # Extract positions for NetworkX
    pos = {node: (drawing.x(idx), drawing.y(idx)) 
           for node, idx in indices.items()}
    
    # Visualize
    plt.figure(figsize=(10, 8))
    nx.draw(nx_graph, pos, node_color='lightblue', 
            node_size=300, with_labels=True)
    # plt.savefig('karate_club.png')
    # plt.show()
    
    print("Visualization created")

.. testoutput:: python

    Visualization created

Customizing Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. testcode:: python

    import networkx as nx
    import egraph as eg
    import matplotlib.pyplot as plt

    nx_graph = nx.karate_club_graph()
    graph = eg.Graph()
    indices = {}
    for node in nx_graph.nodes:
        indices[node] = graph.add_node(node)
    for u, v in nx_graph.edges:
        graph.add_edge(indices[u], indices[v], (u, v))
    
    drawing = eg.DrawingEuclidean2d.initial_placement(graph)
    sm = eg.StressMajorization(graph, drawing, lambda _: 100)
    sm.run(drawing)
    
    pos = {node: (drawing.x(idx), drawing.y(idx)) 
           for node, idx in indices.items()}
    
    # Compute node degrees for sizing
    degrees = dict(nx_graph.degree())
    node_sizes = [degrees[node] * 50 for node in nx_graph.nodes]
    
    # Color by community
    communities = nx.community.greedy_modularity_communities(nx_graph)
    node_colors = []
    for node in nx_graph.nodes:
        for i, comm in enumerate(communities):
            if node in comm:
                node_colors.append(i)
                break
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    nx.draw(nx_graph, pos, 
            node_color=node_colors,
            node_size=node_sizes,
            cmap=plt.cm.Set3,
            with_labels=True,
            font_size=8,
            edge_color='gray',
            alpha=0.7)
    # plt.title('Karate Club Network')
    # plt.savefig('karate_club_styled.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    print("Styled visualization created")

.. testoutput:: python

    Styled visualization created

Choosing the Right Drawing Space
---------------------------------

**Euclidean 2D**
- Most common choice
- Good for general-purpose visualization
- Easy to interpret

**Euclidean nD**
- For high-dimensional data
- Useful for dimensionality reduction
- Can project to 2D/3D for visualization

**Hyperbolic 2D**
- Hierarchical structures
- Trees and DAGs
- Focus+context visualization

**Spherical 2D**
- Global networks
- No preferred direction
- Uniform connectivity

**Torus 2D**
- Periodic structures
- Grid-like networks
- Avoiding edge effects

Best Practices
--------------

1. **Start with Euclidean 2D**: It's the most intuitive and widely supported
2. **Match space to structure**: Use hyperbolic for trees, spherical for global networks
3. **Iterate on layout**: Run algorithms multiple times with different parameters
4. **Visualize incrementally**: Check intermediate results during development
5. **Export high-quality**: Use high DPI for publication-quality figures

Next Steps
----------

* :doc:`../examples/index` - See complete visualization examples
* :doc:`../api/drawing` - Detailed drawing API reference
* :doc:`../getting_started/overview` - Review library capabilities
