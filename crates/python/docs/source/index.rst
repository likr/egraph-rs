Welcome to egraph documentation
===============================

**egraph** is a high-performance, strictly single-threaded Python library for graph visualization and layout algorithms, powered by pure CPU Rust (`abi3-py310`).

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   getting_started/index
   tutorial/index
   api/index
   examples/index

Features
--------

* Zero-copy graph and coordinate buffers (``CsrGraph``, ``Layout2d``, ``LayoutNd``, ``ManifoldLayout2d``)
* Ecosystem interoperability via ``egraph.interop`` (SciPy CSR, NetworkX, igraph)
* Drawing spaces (Euclidean 2D/nD, Poincare Hyperbolic disk, Spherical, 2D Torus)
* SOTA & unified layout solvers:

  * **Track A**: ``TsNetSolver`` (Pure 1+2-Hop & Hybrid Adaptive tsNET with ``ibFFT`` / Barnes-Hut / Exact)
  * **Track B**: ``SgdSolver.sparse_barrier_dog``, ``GuMapSolver``, and 2D ``SpatialIntruderRefiner``
  * **Unified Force & Multilevel Models**: ``TFdpSolver``, ``IbUmapSolver``, ``SgdSolver.snap_t_fdp``, ``SgdSolver.pacmap_graph``, ``ForceAtlas2Solver``, ``YifanHuSolver``
  * **Classical & Post-Processing**: ``StressMajorizationSolver``, ``KamadaKawaiSolver``, ``PrismOverlapRemoval``, ``FdebSolver``, DAG layering, and separation constraints

* Community detection and coarsening (``kmeans_plus_plus``, ``spectral_clustering``, ``louvain_communities``, ``label_propagation``, ``coarsen_graph``)
* Selective quality metrics (``evaluate_metrics``, Scale-Normalized Stress ``SNS``, sweep-line crossings, ``NP1``/``NP2``, ``PRISM`` distortion, Eades 2017 ``S_RNG``/``S_GG`` via ``drng_preservation``/``dgg_preservation``)

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
