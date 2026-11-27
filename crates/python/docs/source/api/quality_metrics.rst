Quality Metrics Module
======================

This module provides selective v6 metrics (``evaluate_metrics`` returning :class:`egraph.MetricReport`), Scale-Normalized Stress ``SNS``, ``geodesic_distortion``, sweep-line and manifold crossings, ``StaticKdTree`` ``NP1``/``NP2``, ``PRISM`` distortion, Eades et al. 2017 unweighted ``S_RNG``/``S_GG`` via ``drng_preservation``/``dgg_preservation``, ``G-CAF``, ``NOR``, ``Sprawlter``, and cluster separation baselines across ``Layout2d``, ``LayoutNd``, and ``ManifoldLayout2d``.

Selective Dispatcher & Report
-----------------------------

.. autoclass:: egraph.MetricReport
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: egraph.evaluate_metrics

Scale-Normalized Stress (SNS) & Geodesic Distortion
---------------------------------------------------

.. autofunction:: egraph.scale_normalized_stress
.. autofunction:: egraph.scale_normalized_stress_detail
.. autofunction:: egraph.sampled_scale_normalized_stress
.. autofunction:: egraph.sampled_scale_normalized_stress_detail
.. autofunction:: egraph.scale_normalized_stress_from_distances
.. autofunction:: egraph.scale_normalized_stress_from_distances_detail
.. autofunction:: egraph.scale_normalized_stress_from_pairs
.. autofunction:: egraph.geodesic_distortion

Crossing Metrics (Euclidean, Torus, Spherical & Hyperbolic)
-----------------------------------------------------------

.. autofunction:: egraph.crossing_number
.. autofunction:: egraph.crossing_angle
.. autofunction:: egraph.crossing_edges
.. autofunction:: egraph.crossing_number_from_pairs
.. autofunction:: egraph.crossing_angle_from_pairs
.. autofunction:: egraph.sweep_crossing_number
.. autofunction:: egraph.sweep_crossing_angle
.. autofunction:: egraph.sampled_crossing_number
.. autofunction:: egraph.crosslessness
.. autofunction:: egraph.crosslessness_from_count
.. autofunction:: egraph.max_crossing_pairs
.. autofunction:: egraph.sampled_crosslessness
.. autofunction:: egraph.torus_crossing_number
.. autofunction:: egraph.crossing_edges_torus
.. autofunction:: egraph.spherical_crossing_number
.. autofunction:: egraph.crossing_edges_spherical
.. autofunction:: egraph.hyperbolic_crossing_number
.. autofunction:: egraph.crossing_edges_hyperbolic
.. autofunction:: egraph.crossing_summary
.. autofunction:: egraph.edge_angle

Neighborhood & Proximity Preservation
-------------------------------------

.. autofunction:: egraph.neighborhood_preservation_1hop
.. autofunction:: egraph.neighborhood_preservation_2hop
.. autofunction:: egraph.sampled_np1
.. autofunction:: egraph.sampled_np2
.. autofunction:: egraph.drng_preservation
.. autofunction:: egraph.dgg_preservation
.. autofunction:: egraph.gabriel_graph_property
.. autofunction:: egraph.gabriel_graph_edges
.. autofunction:: egraph.relative_neighborhood_edges
.. autofunction:: egraph.landmark_stratified_rta
.. autofunction:: egraph.landmark_stratified_rta_detail
.. autofunction:: egraph.landmark_stratified_rta_from_landmarks

Community Spatial Coherence, Overlap & Separation Baselines
-----------------------------------------------------------

.. autofunction:: egraph.adjusted_patch_nig
.. autofunction:: egraph.adjusted_patch_nig_detail
.. autofunction:: egraph.adjusted_patch_nig_multiscale
.. autofunction:: egraph.cluster_area_faithfulness
.. autofunction:: egraph.cluster_area_faithfulness_detail
.. autofunction:: egraph.partition_agreement
.. autofunction:: egraph.distance_consistency
.. autofunction:: egraph.distribution_consistency
.. autofunction:: egraph.class_density_measure
.. autofunction:: egraph.histogram_density_measure
.. autofunction:: egraph.sepme_knng
.. autofunction:: egraph.sepme_rng
.. autofunction:: egraph.sepme_gab
.. autofunction:: egraph.sepme_gong
.. autofunction:: egraph.neighborhood_hit
.. autofunction:: egraph.silhouette_score
.. autofunction:: egraph.calinski_harabasz_score
.. autofunction:: egraph.davies_bouldin_score
.. autofunction:: egraph.cq_kmeans
.. autofunction:: egraph.cq_hac
.. autofunction:: egraph.pairwise_distance_overlap_error
.. autofunction:: egraph.sprawlter_nn
.. autofunction:: egraph.sprawlter_nn_radii
.. autofunction:: egraph.nor_metrics

.. testcode:: python

   coords = [
       [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1],
       [5.0, 0.0], [5.1, 0.0], [5.0, 0.1], [5.1, 0.1],
   ]
   labels = [0, 0, 0, 0, 1, 1, 1, 1]
   layout = eg.Layout2d.from_coords(coords)
   caf = eg.cluster_area_faithfulness(layout, labels)
   detail = eg.cluster_area_faithfulness_detail(layout, labels, return_per_cluster=True)
   assert 0.0 <= caf <= 1.0
   assert detail["num_clusters"] == 2
   assert len(detail["caf_per_cluster"]) == 2

Geometry, Resolution & PRISM Distortion
---------------------------------------

.. autofunction:: egraph.prism_distortion
.. autofunction:: egraph.prism_distortion_with_edges
.. autofunction:: egraph.node_resolution
.. autofunction:: egraph.aspect_ratio
.. autofunction:: egraph.angular_resolution
.. autofunction:: egraph.minimum_angle
.. autofunction:: egraph.minimum_angle_summary
.. autofunction:: egraph.layout_degeneracy
.. autofunction:: egraph.ideal_edge_lengths
.. autofunction:: egraph.ideal_edge_lengths_with_unit
.. autofunction:: egraph.ideal_edge_lengths_scaled
.. autofunction:: egraph.ideal_edge_lengths_from_distances

BFS Distance Helpers
--------------------

.. autofunction:: egraph.all_pairs_bfs
.. autofunction:: egraph.bfs_single_source
