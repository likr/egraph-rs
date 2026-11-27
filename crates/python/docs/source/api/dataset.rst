Dataset Module
==============

Built-in benchmark graph catalog, CSR CSV parsing helpers, and deterministic synthetic graph generators.

.. autoclass:: egraph.DatasetInfo
   :members:
   :undoc-members:
   :show-inheritance:

Catalog & Loaders
-----------------

.. autofunction:: egraph.list_datasets
.. autofunction:: egraph.dataset_catalog
.. autofunction:: egraph.dataset_info
.. autofunction:: egraph.load_dataset
.. autofunction:: egraph.parse_csr_csv

Synthetic Graph Generators
--------------------------

.. autofunction:: egraph.generate_weighted_graph
.. autofunction:: egraph.generate_layered_dag
.. autofunction:: egraph.generate_sbm_graph
.. autofunction:: egraph.generate_torus_grid
.. autofunction:: egraph.generate_spherical_icosphere
.. autofunction:: egraph.generate_spherical_grid
.. autofunction:: egraph.generate_hyperbolic_tree
