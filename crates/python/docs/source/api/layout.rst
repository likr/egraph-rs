Layout Module
=============

This module provides cooperative v6 layout solvers, ``LayoutGenerator`` initializers, and ``LayoutTransform`` passes across 2D (``Layout2d``), 3D (``Layout3d``), $N$-D (``LayoutNd``), non-Euclidean manifolds (``ManifoldLayout2d``), and edge polylines (``EdgePolylines``).

Coarse Backbone & Generators
----------------------------

.. autoclass:: egraph.CoarseBackbone
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: egraph.CoarseBackboneResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: egraph.CoarseBackboneStats
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: egraph.coarse_backbone_init
.. autofunction:: egraph.pacmap_graph_with_backbone
.. autofunction:: egraph.extract_quadrilateral_backbone

.. toctree::
   :maxdepth: 1

   layout/ts_net_solver
   layout/sgd_solver
   layout/embedding_solvers
   layout/gumap_solver
   layout/t_fdp_solver
   layout/ib_umap_solver
   layout/force_atlas2_solver
   layout/fruchterman_reingold_solver
   layout/yifan_hu_solver
   layout/lin_log_solver
   layout/gem_solver
   layout/sel_layout_solver
   layout/backbone_solver
   layout/group_in_a_box_solver
   layout/cluster_refine_solver
   layout/prism_overlap_removal
   layout/forbid_overlap_removal
   layout/g_tree_overlap_removal
   layout/pfs_prime_overlap_removal
   layout/twin_declutterer
   layout/spatial_intruder_refiner
   layout/screened_poisson_field_2d
   layout/screened_poisson_refiner
   layout/fdeb_solver
   layout/stress_majorization_solver
   layout/kamada_kawai_solver
   layout/maxent_stress_solver
   layout/grip_solver
   layout/sfc_layout
