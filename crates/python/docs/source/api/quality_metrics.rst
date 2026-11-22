Quality Metrics Module
======================

This module provides selective v6 metrics (``evaluate_metrics``, Scale-Normalized Stress ``SNS``, sweep-line crossings, ``StaticKdTree`` ``NP1``/``NP2``, ``PRISM`` distortion, ``dRNG``/``dGG``) alongside legacy metric functions.

Selective Dispatcher
--------------------

.. autofunction:: egraph.evaluate_metrics

Scale-Normalized Stress (SNS)
-----------------------------

.. autofunction:: egraph.scale_normalized_stress
.. autofunction:: egraph.sampled_scale_normalized_stress
.. autofunction:: egraph.scale_normalized_stress_from_distances

Crossing Metrics
----------------

.. autofunction:: egraph.sweep_crossing_number
.. autofunction:: egraph.sweep_crossing_angle
.. autofunction:: egraph.sampled_crossing_number
.. autofunction:: egraph.torus_crossing_number
.. autofunction:: egraph.crossing_edges_torus
.. autofunction:: egraph.crossing_summary
.. autofunction:: egraph.edge_angle

Neighborhood & Proximity Preservation
-------------------------------------

.. autofunction:: egraph.neighborhood_preservation_1hop
.. autofunction:: egraph.neighborhood_preservation_2hop_v6
.. autofunction:: egraph.drng_preservation
.. autofunction:: egraph.dgg_preservation
.. autofunction:: egraph.gabriel_graph_property_v6
.. autofunction:: egraph.gabriel_graph_edges
.. autofunction:: egraph.relative_neighborhood_edges

Geometry, Resolution & PRISM Distortion
---------------------------------------

.. autofunction:: egraph.prism_distortion
.. autofunction:: egraph.prism_distortion_with_edges
.. autofunction:: egraph.node_resolution_v6
.. autofunction:: egraph.aspect_ratio_v6
.. autofunction:: egraph.angular_resolution_v6
.. autofunction:: egraph.ideal_edge_lengths_v6
.. autofunction:: egraph.ideal_edge_lengths_with_unit

BFS Distance Helpers
--------------------

.. autofunction:: egraph.all_pairs_bfs
.. autofunction:: egraph.bfs_single_source

Legacy Metrics
--------------

.. autofunction:: egraph.stress
.. autofunction:: egraph.crossing_number
.. autofunction:: egraph.neighborhood_preservation
.. autofunction:: egraph.neighborhood_preservation_2hop
.. autofunction:: egraph.aspect_ratio
.. autofunction:: egraph.angular_resolution
.. autofunction:: egraph.node_resolution
.. autofunction:: egraph.gabriel_graph_property
