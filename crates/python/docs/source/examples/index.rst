Examples
========

This section provides practical examples demonstrating ``egraph`` v6 features.

.. toctree::
   :maxdepth: 2

   track_a_tsnet
   track_b_sparse_barrier
   unified_force_models
   prism_overlap_removal
   fdeb
   manifold_layouts

Overview
--------

Each example is self-contained and verified by Sphinx ``doctest``.

v6 Solvers & Workflows
----------------------

* :doc:`track_a_tsnet` - Track A Pure 1+2-Hop Adaptive tsNET (``TsNetSolver.pure_2hop``) and Row-Normalized ``from_graph_row_normalized`` with ``backend="auto"``
* :doc:`track_b_sparse_barrier` - Track B Regularized DoG 3-Tier Sparse Barrier SGD, ``GuMapSolver``, and ``SpatialIntruderRefiner``
* :doc:`unified_force_models` - ``TFdpSolver``, ``IbUmapSolver``, ``SgdSolver.snap_t_fdp``, ``SgdSolver.pacmap_graph``, ``ForceAtlas2Solver``, and ``YifanHuSolver``
* :doc:`prism_overlap_removal` - $O(N \log N)$ Delaunay PRISM overlap removal and ``prism_distortion``
* :doc:`fdeb` - Cooperative Force-Directed Edge Bundling (``FdebSolver``)
* :doc:`manifold_layouts` - Non-Euclidean ``ManifoldLayout2d`` (Poincaré Hyperbolic, Lorentz Hyperboloid, Spherical, Torus)

Interactive Colab Notebooks
---------------------------

For full end-to-end visual tutorials and benchmarks with ``matplotlib`` plots, see `notebooks/README.md <https://github.com/likr/egraph-rs/blob/master/notebooks/README.md>`_ or launch any of the 9 notebooks directly on Google Colab:

* **01 — Getting Started & Data Structures**: `GitHub <https://github.com/likr/egraph-rs/blob/master/notebooks/01_getting_started_and_data_structures.ipynb>`_ | `Open in Colab <https://colab.research.google.com/github/likr/egraph-rs/blob/master/notebooks/01_getting_started_and_data_structures.ipynb>`_
* **02 — Initialization & Stress Layouts**: `GitHub <https://github.com/likr/egraph-rs/blob/master/notebooks/02_initialization_and_stress_layouts.ipynb>`_ | `Open in Colab <https://colab.research.google.com/github/likr/egraph-rs/blob/master/notebooks/02_initialization_and_stress_layouts.ipynb>`_
* **03 — Neighborhood-Preserving Layouts**: `GitHub <https://github.com/likr/egraph-rs/blob/master/notebooks/03_neighborhood_preserving_layouts.ipynb>`_ | `Open in Colab <https://colab.research.google.com/github/likr/egraph-rs/blob/master/notebooks/03_neighborhood_preserving_layouts.ipynb>`_
* **04 — Force-Directed & Multilevel Layouts**: `GitHub <https://github.com/likr/egraph-rs/blob/master/notebooks/04_force_directed_and_multilevel_layouts.ipynb>`_ | `Open in Colab <https://colab.research.google.com/github/likr/egraph-rs/blob/master/notebooks/04_force_directed_and_multilevel_layouts.ipynb>`_
* **05 — Community, Clustered & DAG Layouts**: `GitHub <https://github.com/likr/egraph-rs/blob/master/notebooks/05_community_clustered_and_dag_layouts.ipynb>`_ | `Open in Colab <https://colab.research.google.com/github/likr/egraph-rs/blob/master/notebooks/05_community_clustered_and_dag_layouts.ipynb>`_
* **06 — Postprocessing, Overlap & Bundling**: `GitHub <https://github.com/likr/egraph-rs/blob/master/notebooks/06_postprocessing_overlap_and_bundling.ipynb>`_ | `Open in Colab <https://colab.research.google.com/github/likr/egraph-rs/blob/master/notebooks/06_postprocessing_overlap_and_bundling.ipynb>`_
* **07 — Non-Euclidean Manifold Layouts**: `GitHub <https://github.com/likr/egraph-rs/blob/master/notebooks/07_non_euclidean_manifold_layouts.ipynb>`_ | `Open in Colab <https://colab.research.google.com/github/likr/egraph-rs/blob/master/notebooks/07_non_euclidean_manifold_layouts.ipynb>`_
* **08 — Quality Metrics & Benchmarks**: `GitHub <https://github.com/likr/egraph-rs/blob/master/notebooks/08_quality_metrics_and_benchmarks.ipynb>`_ | `Open in Colab <https://colab.research.google.com/github/likr/egraph-rs/blob/master/notebooks/08_quality_metrics_and_benchmarks.ipynb>`_
* **09 — Spectral Diffusion & AMG Primitives**: `GitHub <https://github.com/likr/egraph-rs/blob/master/notebooks/09_spectral_diffusion_and_amg_primitives.ipynb>`_ | `Open in Colab <https://colab.research.google.com/github/likr/egraph-rs/blob/master/notebooks/09_spectral_diffusion_and_amg_primitives.ipynb>`_

Quick Example
-------------

.. testcode:: python

    import egraph as eg

    # Create a 4x4 grid graph via CsrGraph.from_edges
    edges = []
    for r in range(4):
        for c in range(4):
            u = r * 4 + c
            if c + 1 < 4:
                edges.append((u, u + 1))
            if r + 1 < 4:
                edges.append((u, u + 4))

    graph = eg.CsrGraph.from_edges(16, edges)
    layout = eg.pivot_mds_init(graph, num_pivots=8, seed=42)
    eg.TsNetSolver.pure_2hop(
        graph,
        backend="auto",
        stage_iterations=(20, 20, 30),
        seed=42,
    ).run(layout)

    report = eg.evaluate_metrics(graph, layout, ["sns", "np2", "crossing_number"])
    assert layout.is_all_finite()
    assert 0.0 <= report["neighborhood_preservation_2hop"] <= 1.0
    print("Quick example complete")

.. testoutput:: python

    Quick example complete
