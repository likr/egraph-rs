Clustering Module
=================

Pure-Rust single-threaded community detection, graph coarsening, and clustered layout utilities returning typed zero-copy result objects.

Result Classes
--------------

.. autoclass:: egraph.LouvainResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: egraph.LeidenResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: egraph.KMeansResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: egraph.LabelPropagationResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: egraph.SpectralClusteringResult
   :members:
   :undoc-members:
   :show-inheritance:

Community Detection & Coarsening Functions
------------------------------------------

.. autofunction:: egraph.kmeans_plus_plus
.. autofunction:: egraph.spectral_clustering
.. autofunction:: egraph.louvain_communities
.. autofunction:: egraph.leiden_communities
.. autofunction:: egraph.louvain_coarsen
.. autofunction:: egraph.label_propagation
.. autofunction:: egraph.coarsen_graph
.. autofunction:: egraph.project_clustered_layout
.. autofunction:: egraph.project_clustered_layout_with_radii
.. autofunction:: egraph.align_clustered_centroids
.. autofunction:: egraph.refine_cluster_layout
.. autofunction:: egraph.squarified_treemap

Example
~~~~~~~

.. testcode:: python

   import egraph as eg

   edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
   g = eg.CsrGraph.from_edges(6, edges, directed=False, sanitize=True)
   res = eg.leiden_communities(g, seed=42)
   assert res.labels.shape == (6,) and res.num_communities >= 1

   h = eg.louvain_coarsen(g, max_coarse_nodes=3, seed=42)
   assert h["num_clusters"] <= 3

   bb = eg.coarse_backbone_init(g, max_coarse_nodes=3, coarse_t_max=10, seed=42)
   assert bb["fine_layout"].num_nodes == 6
   print("Clustering example complete")

.. testoutput:: python

   Clustering example complete
