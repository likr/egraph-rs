SgdSolver
=========

Cooperative ``u32`` SoA Stochastic Gradient Descent solver supporting Track B 3-tier Sparse Barrier SGD (``sparse_barrier_dog``, ``sparse_barrier_from_landmarks``), ``ortmann_sparse_stress``, ``largevis``, ``drgraph``, ``graph_cne``, ``snap_t_fdp``, ``pacmap_graph``, diagonal-preconditioned mini-batch stress ``sgd2`` (distinct from Ahmed et al. 2021 multi-criteria ``(SGD)^2``), ``full``, ``random_pair``, ``omega``, ``streaming_negatives``, custom keyword-only pair constructors (``from_pairs``, ``from_pairs_with_hinges``, ``from_packed_symmetric``), and schedule-window stepping (``reset_schedule``, ``run_schedule_window``) across ``Layout2d``, ``LayoutNd``, and ``ManifoldLayout2d`` (``supports_manifold``).

.. autoclass:: egraph.SgdSolver
   :members:
   :undoc-members:
   :show-inheritance:
