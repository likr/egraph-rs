Linear Algebra & Neighborhood Module
====================================

Pure-Rust linear algebra, Chebyshev heat/DoG diffusion, spectral initialization, and $1+2$-hop / adaptive $k$-NN neighborhood primitives.

Data Containers
---------------

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

Initialization, Spectral & Chebyshev Primitives
-----------------------------------------------

.. autofunction:: egraph.select_maxmin_sp_pivots
.. autofunction:: egraph.pivot_mds_init
.. autofunction:: egraph.koren_spectral_init
.. autofunction:: egraph.classical_mds_2d
.. autofunction:: egraph.rd_mds
.. autofunction:: egraph.normalize_layout_2d
.. autofunction:: egraph.representative_mds
.. autofunction:: egraph.smallest_laplacian_eigenpairs
.. autofunction:: egraph.spectral_moment_stats
.. autofunction:: egraph.chebyshev_heat_spmv
.. autofunction:: egraph.chebyshev_dog_jl_embedding
.. autofunction:: egraph.compute_resistance_embedding
.. autofunction:: egraph.compute_diffusion_embedding

Neighborhood & Probability Builders
-----------------------------------

.. autofunction:: egraph.extract_1hop_2hop_csr
.. autofunction:: egraph.extract_sparse_1hop_2hop_pairs
.. autofunction:: egraph.build_pure_2hop_knn
.. autofunction:: egraph.compute_2hop_neighborhood_sizes
.. autofunction:: egraph.compute_nodewise_2hop_perplexities
.. autofunction:: egraph.compute_median_2hop_perplexity
.. autofunction:: egraph.build_hybrid_adaptive_knn
.. autofunction:: egraph.build_rerank_knn
.. autofunction:: egraph.partial_bfs_knn
.. autofunction:: egraph.compute_knn_probabilities
.. autofunction:: egraph.graph_row_normalized_probabilities
.. autofunction:: egraph.symmetrize_row_normalized
.. autofunction:: egraph.compute_fuzzy_simplicial_set
.. autofunction:: egraph.find_ab_params
.. autofunction:: egraph.compute_horvitz_thompson_gamma
.. autofunction:: egraph.compute_regularized_dog_varadhan_landmarks
.. autofunction:: egraph.build_gumap_12hop_priority_csr
.. autofunction:: egraph.apply_relative_distance_floor
