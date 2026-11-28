SgdSolver & Sgd2Solver
======================

Cooperative ``u32`` SoA Stochastic Gradient Descent stress and barrier solvers supporting Track B 3-tier Sparse Barrier SGD (``sparse_barrier_dog``, ``sparse_barrier_from_landmarks``), ``ortmann_sparse_stress``, ``full``, ``random_pair``, ``omega``, ``streaming_negatives``, custom keyword-only pair constructors (``from_pairs``, ``from_pairs_with_hinges``, ``from_packed_symmetric``), and schedule-window stepping (``reset_schedule``, ``run_schedule_window``) across explicit ``target="layout2d" | "layout3d" | "layout_nd" | "manifold2d"`` (``supports_manifold``), alongside separated diagonal-preconditioned mini-batch :class:`egraph.Sgd2Solver`.

.. autoclass:: egraph.SgdSolver
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: egraph.Sgd2Solver
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: egraph.SparseBarrierStats
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: egraph.DogVaradhanStats
   :members:
   :undoc-members:
   :show-inheritance:
