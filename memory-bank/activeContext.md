# Active Context: egraph-rs

## Current Work Focus

The current focus is on the layout algorithms in the graph visualization system, particularly:

1. **Stochastic Gradient Descent (SGD)** implementations

   - Full SGD (all-pairs shortest path distances)
   - Sparse SGD (pivot-based sparse approximation)
   - Distance-adjusted SGD (dynamic distance adjustment)
   - Various learning rate schedulers:
     - Constant: Maintains a fixed learning rate
     - Linear: Linear decay of learning rate
     - Exponential: Exponential decay of learning rate
     - Quadratic: Quadratic decay of learning rate
     - Reciprocal: Reciprocal decay of learning rate
   - Based on research: Zheng, J. X., Pawar, S., & Goodman, D. F. (2018). "Graph drawing by stochastic gradient descent"

2. **Stress Majorization** algorithm

   - Iterative stress minimization
   - Conjugate gradient solver
   - Based on research: Gansner et al. (2004) "Graph drawing by stress majorization"

3. **Multidimensional Scaling (MDS)** implementations

   - Classical MDS: Standard implementation that computes a full distance matrix
   - Pivot MDS: Efficient implementation that uses a subset of nodes as pivots
   - Uses eigendecomposition and double centering to transform distance matrices
   - Based on research: Cox, T. F., & Cox, M. A. (2000). Multidimensional scaling. Chapman & Hall/CRC.

4. **Overlap Removal** techniques for improving readability

5. **Quality Metrics** for evaluating layout effectiveness
   - Stress: How well layout preserves graph-theoretical distances
   - Ideal Edge Lengths: How well edge lengths match their ideal lengths
   - Neighborhood Preservation: How well the layout preserves local neighborhoods
   - Crossing Number: Count of edge crossings in the layout
   - Edge Angle: Angles at which edges cross
   - Aspect Ratio: Balance between width and height of the drawing
   - Angular Resolution: Angles between edges connected to the same node
   - Node Resolution: How well nodes are distributed in the drawing space
   - Gabriel Graph Property: Adherence to the Gabriel graph condition

## Recent Changes

- Refined various layout algorithm implementations
- Improved documentations of drawing implementations for different geometric spaces:
  - Euclidean 2D: 2D Cartesian coordinate drawings
  - Spherical 2D: Drawings on a spherical surface
  - Hyperbolic 2D: Drawings in hyperbolic space
  - Torus 2D: Drawings on a torus surface
  - N-dimensional Euclidean: Drawings in higher-dimensional Euclidean spaces
- Enhanced WebAssembly bindings with improved JavaScript API features:
  - Consistent naming conventions (camelCase for JavaScript)
  - JSDoc-style comments for better developer experience
  - Transparent data handling between Rust and JavaScript
  - Memory safety improvements
  - Method chaining support
  - Robust error handling
  - JavaScript callback support

## Next Steps

1. **Layout Algorithm Refinement**:

   - Fine-tune SGD schedulers for better convergence
   - Optimize stress majorization for large graphs
   - Improve performance of overlap removal
   - Address performance issues with large graphs (>10,000 nodes)

2. **Integration Improvements**:

   - Ensure consistent behavior across drawing implementations
   - Improve interoperability between layout algorithms
   - Resolve inconsistencies between language bindings (Rust, Python, JavaScript)

3. **Documentation and Examples**:

   - Add examples showcasing different layout algorithms
   - Document best practices for selecting appropriate layout algorithms
   - Create comprehensive API documentation
   - Develop tutorials for common use cases

4. **Testing Enhancements**:
   - Expand test coverage
   - Implement performance benchmarks
   - Validate cross-platform behavior consistency

## Active Decisions and Considerations

1. **Performance Trade-offs**:

   - Balance between layout quality and computation speed
   - Memory usage optimization for large graphs
   - Addressing high memory consumption for dense graphs in WebAssembly context

2. **API Design**:

   - Ensure consistent interfaces across layout implementations
   - Consider future extensibility for new layout algorithms
   - Maintain API stability while allowing for improvements

3. **Cross-platform Consistency**:
   - Validate behavior consistency across Rust, WebAssembly, and Python
   - Ensure uniform error handling across language boundaries
   - Maintain consistent naming conventions appropriate to each language
