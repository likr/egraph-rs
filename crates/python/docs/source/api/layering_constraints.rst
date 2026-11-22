Layering, Separation Constraints & Spatial Utilities
====================================================

Functions and classes for DAG cycle removal, layer assignment, 1D/2D separation constraint projection, rectangle overlap projection, and Delaunay triangulation.

Layering & DAG Utilities
------------------------

.. autofunction:: egraph.cycle_edges
.. autofunction:: egraph.remove_cycles
.. autofunction:: egraph.assign_layers
.. autofunction:: egraph.generate_layered_constraints

Separation Constraints & Projection
-----------------------------------

.. autoclass:: egraph.SeparationConstraint
   :members:
   :undoc-members:

.. autofunction:: egraph.project_1d
.. autofunction:: egraph.project_layout_axis
.. autofunction:: egraph.project_rectangle_no_overlap_2d
.. autofunction:: egraph.project_clustered_rectangle_no_overlap_2d

Spatial Triangulation & Overlap Detection
-----------------------------------------

.. autofunction:: egraph.delaunay_triangulation
.. autofunction:: egraph.delaunay_to_csr
.. autofunction:: egraph.rectangle_overlaps
