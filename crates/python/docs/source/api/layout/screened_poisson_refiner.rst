ScreenedPoissonRefiner
======================

Cooperative Screened Poisson unilateral upper-bound density field refiner supporting ``step()``, ``run_steps()``, ``run()``, and ``apply_epoch_step()``.

.. autoclass:: egraph.ScreenedPoissonRefiner
   :members:
   :undoc-members:
   :show-inheritance:

Example
-------

.. code-block:: python

   >>> import numpy as np
   >>> coords = np.array([[0.0, 0.0], [0.05, 0.02], [-0.04, 0.03], [0.02, -0.05]])
   >>> layout = eg.Layout2d.from_coords(coords)
   >>> refiner = eg.ScreenedPoissonRefiner(4, grid_size=64, iterations=3, rho_max=0.2)
   >>> status = refiner.run(layout)
   >>> status in ("running", "converged", "completed")
   True
