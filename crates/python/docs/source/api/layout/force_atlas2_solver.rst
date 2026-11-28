ForceAtlas2Solver
=================

Cooperative ForceAtlas2 continuous force-directed placement solver supporting ``target="layout2d" | "layout3d" | "layout_nd"`` with ``backend="barnes_hut" | "exact"`` (``"auto"`` resolves to ``"barnes_hut"`` on 2D/3D and ``"exact"`` on $N$-D) and ForceAtlas2 logarithmic attraction mode (``linlog_mode``, Jacomy et al. 2014, distinct from Noack 2007 true LinLog ``LinLogSolver``).

.. autoclass:: egraph.ForceAtlas2Solver
   :members:
   :undoc-members:
   :show-inheritance:
