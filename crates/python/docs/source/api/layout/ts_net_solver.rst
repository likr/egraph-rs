TsNetSolver
===========

Unified Track A tsNET solver supporting Pure 1+2-Hop adaptive perplexity (``pure_2hop``), direct row-normalized ``A1 + w2 * A2`` affinity (``from_graph_row_normalized``), custom $k$-NN (``from_knn``), sparse probabilities (``from_probabilities``), custom multi-stage schedules (``from_probabilities_with_stages`` with :class:`egraph.TsNetStage`), and partial-BFS initialization across explicit ``target="layout2d" | "layout3d" | "layout_nd" | "manifold2d"`` and Option 1 backends (``"auto"``, ``"fit_fft"``, ``"linear_fft"``, ``"barnes_hut"``, ``"exact"``; inspect via ``resolved_backend``).

.. autoclass:: egraph.TsNetSolver
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: egraph.TsNetStage
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: egraph.FitFieldStats
   :members:
   :undoc-members:
   :show-inheritance:
