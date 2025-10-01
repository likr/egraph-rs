Overview
========

egraph is a high-performance Python library for graph visualization and layout algorithms, powered by a Rust backend for exceptional speed and efficiency.

What is egraph?
---------------

egraph provides a comprehensive toolkit for:

* **Creating and manipulating graphs** - Build graph structures with nodes and edges
* **Computing optimal layouts** - Apply various algorithms to position nodes aesthetically
* **Visualizing in multiple spaces** - Support for Euclidean, Hyperbolic, Spherical, and Torus geometries
* **Evaluating layout quality** - Measure and optimize visualization effectiveness

Key Features
------------

High Performance
^^^^^^^^^^^^^^^^

egraph is built on a Rust core, providing:

* **Fast computation** - Rust's zero-cost abstractions and memory safety
* **Efficient algorithms** - Optimized implementations of graph layout algorithms
* **Low memory footprint** - Careful memory management for large graphs

Multiple Layout Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

egraph supports a variety of layout algorithms:

* **SGD (Stochastic Gradient Descent)** - Fast, scalable force-directed layout
  
  * FullSgd - Complete pairwise distance optimization
  * SparseSgd - Efficient layout for large graphs using pivot nodes
  * Omega - Spectral coordinates-based SGD

* **Stress Majorization** - Iterative optimization for high-quality layouts
* **MDS (Multidimensional Scaling)** - Distance-preserving dimensionality reduction
  
  * ClassicalMds - Eigendecomposition-based approach
  * PivotMds - Scalable variant using landmark nodes

* **Kamada-Kawai** - Spring-based energy minimization
* **Overlap Removal** - Eliminate node overlaps while preserving layout structure

Multiple Drawing Spaces
^^^^^^^^^^^^^^^^^^^^^^^^

Visualize graphs in different geometric spaces:

* **Euclidean** - Standard 2D and n-dimensional layouts
* **Hyperbolic** - Layouts on hyperbolic plane (Poincaré disk model)
* **Spherical** - Layouts on sphere surface
* **Torus** - Layouts on torus with periodic boundaries

Quality Metrics
^^^^^^^^^^^^^^^

Evaluate and optimize your layouts with built-in metrics:

* **Edge Crossings** - Count and minimize edge intersections
* **Angular Resolution** - Measure angle distribution at nodes
* **Aspect Ratio** - Evaluate layout shape and proportions
* **Neighborhood Preservation** - Assess how well local structure is maintained
* **Gabriel Graph Property** - Measure proximity graph properties

NetworkX Integration
^^^^^^^^^^^^^^^^^^^^

Seamlessly work with NetworkX graphs:

* Convert between NetworkX and egraph formats
* Use NetworkX's rich graph creation and analysis tools
* Visualize with matplotlib using egraph layouts

Use Cases
---------

egraph is ideal for:

* **Social Network Analysis** - Visualize relationships and communities
* **Biological Networks** - Display protein interactions, gene networks
* **Knowledge Graphs** - Represent and explore semantic relationships
* **Software Architecture** - Visualize dependencies and call graphs
* **Transportation Networks** - Display routes and connections
* **Research and Education** - Study graph theory and algorithms

Architecture
------------

egraph follows a modular architecture:

.. code-block:: text

    ┌─────────────────────────────────────┐
    │         Python Interface            │
    │    (PyO3 bindings to Rust)          │
    ├─────────────────────────────────────┤
    │         Rust Core Library           │
    │                                     │
    │  ┌──────────┐  ┌─────────────┐     │
    │  │  Graph   │  │   Drawing   │     │
    │  │  Types   │  │   Spaces    │     │
    │  └──────────┘  └─────────────┘     │
    │                                     │
    │  ┌──────────┐  ┌─────────────┐     │
    │  │  Layout  │  │   Quality   │     │
    │  │Algorithm │  │   Metrics   │     │
    │  └──────────┘  └─────────────┘     │
    └─────────────────────────────────────┘

This architecture provides:

* **Type safety** - Rust's strong type system prevents common errors
* **Memory safety** - No segmentation faults or memory leaks
* **Python convenience** - Pythonic API with familiar patterns
* **Performance** - Near-native speed for computationally intensive operations

Philosophy
----------

egraph is designed with these principles:

1. **Performance First** - Leverage Rust for maximum speed
2. **Ease of Use** - Pythonic API that feels natural
3. **Flexibility** - Support multiple algorithms and spaces
4. **Quality** - Provide tools to measure and improve layouts
5. **Interoperability** - Work well with existing Python ecosystem

Next Steps
----------

* :doc:`installation` - Install egraph and get started
* :doc:`quickstart` - Create your first graph layout
* :doc:`../tutorial/01_graph_basics` - Learn graph manipulation in depth
