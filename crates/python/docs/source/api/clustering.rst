Clustering Module
=================

Pure-Rust single-threaded clustering and graph coarsening functions.

.. autofunction:: egraph.kmeans_plus_plus
.. autofunction:: egraph.spectral_clustering
.. autofunction:: egraph.louvain_communities
.. autofunction:: egraph.louvain_coarsen
.. autofunction:: egraph.label_propagation
.. autofunction:: egraph.coarsen_graph
.. autofunction:: egraph.project_clustered_layout
.. autofunction:: egraph.project_clustered_layout_with_radii
.. autofunction:: egraph.align_clustered_centroids

Example
~~~~~~~

.. code-block:: python

   >>> import egraph as eg
   >>> edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
   >>> g = eg.CsrGraph.from_edges(6, edges, directed=False, sanitize=True)
   >>> h = eg.louvain_coarsen(g, max_coarse_nodes=3, seed=42)
   >>> h["num_clusters"] <= 3
   True
   >>> bb = eg.coarse_backbone_init(g, max_coarse_nodes=3, coarse_t_max=10, seed=42)
   >>> bb["fine_layout"].num_nodes == 6
   True
