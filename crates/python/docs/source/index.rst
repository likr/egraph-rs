Welcome to egraph documentation
===============================

**egraph** is a high-performance, strictly single-threaded Python library for graph visualization and layout algorithms, powered by pure CPU Rust (``abi3-py310``).

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
* 3D and $N$-dimensional spatial engine (``ArenaOctree3d``, ``DynamicKdTree``, 11 $N$-D solvers, $N$-D quality metrics)
* Non-Euclidean manifold engine (Poincaré hyperbolic disk ``"hyperbolic"``, Lorentz hyperboloid ``"hyperbolic_lorentz"``, sphere ``"spherical"``, periodic torus ``"torus"`` with ``ManifoldMetric``, ``torus_spectral_phase_init``, ``torus_auto_pan``, 6 manifold solvers, and geodesic crossings)
* Ecosystem interoperability via ``egraph.interop`` (``scipy.sparse``, ``networkx``, ``igraph`` via ``to_igraph`` / ``to_networkx``)
* SOTA & unified layout solvers:

  * **Track A**: ``TsNetSolver`` (Pure 1+2-Hop ``pure_2hop`` & row-normalized ``from_graph_row_normalized`` with ``backend="auto"`` / ``"fft"`` / ``"linear_fft"`` / ``"barnes_hut"`` / ``"exact"``)
  * **Track B**: ``SgdSolver.sparse_barrier_dog``, ``GuMapSolver``, and 2D ``SpatialIntruderRefiner``
  * **Unified Force & Multilevel Models**: ``TFdpSolver``, ``IbUmapSolver``, ``SgdSolver.snap_t_fdp``, ``SgdSolver.pacmap_graph``, ``ForceAtlas2Solver``, ``YifanHuSolver``, ``MaxentStressSolver``, ``GripSolver``, ``GemSolver``, ``SelLayoutSolver``, ``SfcLayoutSolver``, ``BackboneSolver``, ``GroupInABoxSolver``
  * **Classical, Overlap & Post-Processing**: ``StressMajorizationSolver``, ``KamadaKawaiSolver``, ``PrismOverlapRemoval``, ``ForbidOverlapRemoval``, ``GTreeOverlapRemoval``, ``PfsPrimeOverlapRemoval``, ``TwinDeclutterer``, ``ScreenedPoissonField2d`` / ``ScreenedPoissonRefiner``, decoupled ``FdebSolver``, ``transitive_reduction``, DAG layering, and separation constraints

* Community detection (5 zero-copy Result classes: ``LouvainResult``, ``LeidenResult``, ``KMeansResult``, ``LabelPropagationResult``, ``SpectralClusteringResult``) and coarsening (``coarsen_graph``, ``coarse_backbone_init``)
* Selective quality metrics returning ``MetricReport`` (``evaluate_metrics``, Scale-Normalized Stress ``SNS``, ``geodesic_distortion``, sweep-line & manifold crossings, ``NP1``/``NP2``, ``PRISM`` distortion, ``G-CAF``, ``NOR``, ``sprawlter_nn``, Eades 2017 ``S_RNG``/``S_GG`` via ``drng_preservation``/``dgg_preservation``, and cluster separation baselines)
* Built-in benchmark catalog (``list_datasets``, ``dataset_catalog``), 7 synthetic graph generators, ``DijkstraWorkspace``, MDS, Laplacian builders, ``ChebyshevDogKernel``, and the companion ``egraph-cli`` tool

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
