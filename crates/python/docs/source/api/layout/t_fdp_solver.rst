TFdpSolver
==========

Cooperative bounded Student-$t$ force-directed placement (``t-FDP``) solver supporting ``target="layout2d" | "layout3d" | "layout_nd"`` with ``backend="ib_fft" | "barnes_hut" | "exact"`` (``"auto"`` resolves to ``"ib_fft"`` on 2D, ``"barnes_hut"`` on 3D, and ``"exact"`` on $N$-D).

.. autoclass:: egraph.TFdpSolver
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: egraph.FieldFftStats
   :members:
   :undoc-members:
   :show-inheritance:
