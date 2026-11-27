Linear Algebra & Neighborhood Module
====================================

Pure-Rust linear algebra, shortest paths (``DijkstraWorkspace``), Chebyshev heat/DoG diffusion (``ChebyshevDogKernel``), spectral initialization, MDS, AMG-PCG, and $1+2$-hop / adaptive $k$-NN neighborhood primitives.

Data Containers & Workspaces
----------------------------

.. autoclass:: egraph.KnnGraph
   :members:
   :undoc-members:

.. autoclass:: egraph.SparseProbabilities
   :members:
   :undoc-members:

.. autoclass:: egraph.Sparse12HopPairs
   :members:
   :undoc-members:

.. autoclass:: egraph.DogVaradhanLandmarks
   :members:
   :undoc-members:

.. autoclass:: egraph.RhghLandmarks
   :members:
   :undoc-members:

.. autoclass:: egraph.CsrMatrix
   :members:
   :undoc-members:

.. autoclass:: egraph.ChebyshevDogKernel
   :members:
   :undoc-members:

.. autoclass:: egraph.DijkstraWorkspace
   :members:
   :undoc-members:

.. autoclass:: egraph.AmgPcgSolver
   :members:
   :undoc-members:

.. autoclass:: egraph.EffectiveResistanceResult
   :members:
   :undoc-members:

.. autoclass:: egraph.ResistanceCurvatureFlowResult
   :members:
   :undoc-members:

Shortest Paths, Laplacians, MDS & Spectral Primitives
-----------------------------------------------------

.. autofunction:: egraph.dijkstra_single_source
.. autofunction:: egraph.all_pairs_dijkstra
.. autofunction:: egraph.laplacian
.. autofunction:: egraph.combinatorial_laplacian
.. autofunction:: egraph.symmetric_normalized_laplacian
.. autofunction:: egraph.random_walk_laplacian
.. autofunction:: egraph.shifted_normalized_laplacian
.. autofunction:: egraph.select_maxmin_sp_pivots
.. autofunction:: egraph.pivot_mds
.. autofunction:: egraph.pivot_mds_init
.. autofunction:: egraph.pivot_mds_from_distances
.. autofunction:: egraph.koren_spectral_init
.. autofunction:: egraph.classical_mds
.. autofunction:: egraph.classical_mds_2d
.. autofunction:: egraph.classical_mds_from_distances
.. autofunction:: egraph.hde
.. autofunction:: egraph.hde_nd
.. autofunction:: egraph.rd_mds
.. autofunction:: egraph.normalize_layout_2d
.. autofunction:: egraph.robust_normalize_layout_2d
.. autofunction:: egraph.representative_mds
.. autofunction:: egraph.smallest_laplacian_eigenpairs
.. autofunction:: egraph.symmetric_eigen_jacobi
.. autofunction:: egraph.spectral_moment_stats
.. autofunction:: egraph.anisotropic_laplacian
.. autofunction:: egraph.build_dog_terms
.. autofunction:: egraph.build_regularized_dog_terms
.. autofunction:: egraph.adaptive_chebyshev_degree
.. autofunction:: egraph.adaptive_chebyshev_degree_scaled
.. autofunction:: egraph.compute_chebyshev_bandpass_coefficients
.. autofunction:: egraph.project_psd_spectral_filter
.. autofunction:: egraph.chebyshev_heat_spmv
.. autofunction:: egraph.chebyshev_heat_spmv_matrix
.. autofunction:: egraph.chebyshev_dog_jl_embedding
.. autofunction:: egraph.chebyshev_dog_jl_embedding_matrix
.. autofunction:: egraph.compute_resistance_embedding
.. autofunction:: egraph.compute_diffusion_embedding
.. autofunction:: egraph.compute_effective_resistances
.. autofunction:: egraph.resistance_curvature_flow

Example
~~~~~~~

.. testcode:: python

   import egraph as eg

   edges = [(u, v) for u in range(4) for v in range(u + 1, 4)]
   edges += [(u, v) for u in range(4, 8) for v in range(u + 1, 8)] + [(3, 4)]
   g = eg.CsrGraph.from_edges(8, edges, directed=False, sanitize=True)
   er = eg.compute_effective_resistances(g)
   assert er.foster_rel_err < 1e-10
   flow = eg.resistance_curvature_flow(g, steps=2, eta=0.2)
   assert len(flow.step_history()) == 3
   lg = flow.length_graph()
   assert lg.num_nodes == 8 and lg.is_weighted
   print("Effective resistance & RCF ok")

.. testoutput:: python

   Effective resistance & RCF ok

Neighborhood & Probability Builders
-----------------------------------

.. autofunction:: egraph.extract_1hop_2hop_csr
.. autofunction:: egraph.extract_sparse_1hop_2hop_pairs
.. autofunction:: egraph.build_pure_2hop_knn
.. autofunction:: egraph.build_capped_12hop_knn
.. autofunction:: egraph.compute_2hop_neighborhood_sizes
.. autofunction:: egraph.compute_nodewise_2hop_perplexities
.. autofunction:: egraph.compute_median_2hop_perplexity
.. autofunction:: egraph.build_hybrid_adaptive_knn
.. autofunction:: egraph.build_rerank_knn
.. autofunction:: egraph.partial_bfs_knn
.. autofunction:: egraph.compute_knn_probabilities
.. autofunction:: egraph.graph_row_normalized_probabilities
.. autofunction:: egraph.graph_12hop_row_normalized_probabilities
.. autofunction:: egraph.symmetrize_row_normalized
.. autofunction:: egraph.compute_fuzzy_simplicial_set
.. autofunction:: egraph.find_ab_params
.. autofunction:: egraph.compute_horvitz_thompson_gamma
.. autofunction:: egraph.compute_pivoted_rhgh_landmarks
.. autofunction:: egraph.compute_regularized_dog_varadhan_landmarks
.. autofunction:: egraph.compute_regularized_dog_varadhan_landmarks_from_matrix
.. autofunction:: egraph.build_gumap_12hop_priority_csr
.. autofunction:: egraph.apply_relative_distance_floor
