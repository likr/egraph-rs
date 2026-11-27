FdebSolver
==========

Cooperative Force-Directed Edge Bundling (FDEB) solver constructed from a ``CsrGraph`` (``FdebSolver(graph, ...)``) and stepped polymorphically on ``Layout2d`` or ``LayoutNd`` via ``step(layout)`` / ``run_steps(layout, steps)`` / ``run(layout)``, returning flat ``edge_offsets()``, ``points()``, ``points_axis()``, and ``points_z()`` NumPy coordinate arrays.

.. autoclass:: egraph.FdebSolver
   :members:
   :undoc-members:
   :show-inheritance:
