ManifoldLayout2d
================

2D non-Euclidean manifold coordinate container supporting Poincaré hyperbolic disk (``"hyperbolic"``), Lorentz hyperboloid (``"hyperbolic_lorentz"`` with ``to_poincare_layout()``), unit sphere $S^2$ (``"spherical"``), and periodic 2D torus $T^2$ (``"torus"`` with ``torus_spectral_phase_init()`` and ``torus_auto_pan()``), plus physical ``ManifoldMetric`` scaling (``scale``, ``aspect``, ``metric()``, ``set_scale()``, ``set_aspect()``, ``set_torus_extents()``, ``rescale_geodesic()``).

.. autoclass:: egraph.ManifoldLayout2d
   :members:
   :undoc-members:
   :show-inheritance:

Hyperbolic, Spherical & Torus Geodesic Helpers
----------------------------------------------

.. autofunction:: egraph.poincare_distance
.. autofunction:: egraph.poincare_mobius_add
.. autofunction:: egraph.poincare_einstein_midpoint
.. autofunction:: egraph.lorentz_distance
.. autofunction:: egraph.lorentz_to_poincare_point
.. autofunction:: egraph.lorentz_exp_map
.. autofunction:: egraph.lorentz_log_map
.. autofunction:: egraph.lorentz_einstein_midpoint
.. autofunction:: egraph.spherical_distance
.. autofunction:: egraph.torus_distance
.. autofunction:: egraph.wrap_torus_point
.. autofunction:: egraph.torus_edge_segments
.. autofunction:: egraph.torus_layout_edge_segments
