ScreenedPoissonField2d
======================

Reusable single-precision ``f32`` R2C-FFT Screened Poisson unilateral upper-bound 2D density field evaluator.

.. autoclass:: egraph.ScreenedPoissonField2d
   :members:
   :undoc-members:
   :show-inheritance:

Example
-------

.. code-block:: python

   >>> import numpy as np
   >>> coords = np.array([[0.0, 0.0], [0.05, 0.02], [-0.04, 0.03], [0.02, -0.05]])
   >>> layout = eg.Layout2d.from_coords(coords)
   >>> field = eg.ScreenedPoissonField2d(grid_size=64, rho_max=0.2)
   >>> res = field.evaluate(layout)
   >>> res["fft_2d_calls"] == 6 and res["energy"] > 0.0
   True
