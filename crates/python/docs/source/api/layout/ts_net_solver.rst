TsNetSolver
===========

Unified Track A tsNET solver supporting Pure 1+2-Hop adaptive perplexity (``pure_2hop``), direct row-normalized ``A1 + w2 * A2`` affinity (``from_graph_row_normalized``, defaulting to ``mode="two_step_random_walk"``), custom $k$-NN (``from_knn``), sparse probabilities (``from_probabilities``), custom multi-stage schedules (``from_probabilities_with_stages`` with :class:`egraph.TsNetStage`), and partial-BFS initialization across ``"auto"`` (``N <= 256`` Exact, ``> 256`` ``LinearFft``; inspect via ``resolved_backend``), ``"fft"``, ``"linear_fft"``, ``"barnes_hut"``, and ``"exact"`` backends, operating polymorphically on ``Layout2d``, ``LayoutNd``, and ``ManifoldLayout2d``.

.. autoclass:: egraph.TsNetSolver
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: egraph.TsNetStage
   :members:
   :undoc-members:
   :show-inheritance:
