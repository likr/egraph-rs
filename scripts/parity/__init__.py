"""Domain-scoped parity suite package for egraph Python and WASM targets."""

from __future__ import annotations

import pathlib
from typing import Any

from parity.py_clustering_solvers import run_clustering_solvers_python_checks
from parity.py_community_baselines import run_community_baselines_python_checks
from parity.py_core_and_metrics import (
    run_core_csr_layout_checks,
    run_geometry_metrics_spatial_checks,
)
from parity.py_embedding_classic_solvers import (
    run_embedding_classic_solvers_python_checks,
)
from parity.py_force_overlap_layering import run_force_overlap_layering_python_checks
from parity.py_gcaf import run_gcaf_python_checks
from parity.py_linalg_neighborhood import run_linalg_neighborhood_python_checks
from parity.wasm_node_metrics_and_community import run_wasm_node_parity
from parity.xtarget_classic_and_overlap_baselines import (
    run_cross_target_classic_overlap_baselines_parity,
)
from parity.xtarget_clustering_and_manifolds import (
    run_cross_target_clustering_manifold_parity,
)
from parity.xtarget_community_and_backbone_solvers import (
    run_cross_target_community_backbone_parity,
)
from parity.xtarget_core_dataset_manifold import (
    run_cross_target_core_dataset_manifold_parity,
)
from parity.xtarget_core_solvers import run_cross_target_core_solvers_parity
from parity.xtarget_embeddings_and_knn_probs import (
    run_cross_target_embeddings_knn_probs_parity,
)
from parity.xtarget_error_contracts_and_guards import (
    run_cross_target_error_contracts_parity,
)
from parity.xtarget_force_and_multilevel import run_cross_target_force_multilevel_parity
from parity.xtarget_gcaf import run_cross_target_gcaf_parity
from parity.xtarget_hde_and_sgd_presets import run_cross_target_hde_sgd_presets_parity
from parity.xtarget_layout3d_and_generators import (
    run_cross_target_layout3d_generators_parity,
)
from parity.xtarget_linalg_spectral_amg import run_cross_target_linalg_spectral_parity
from parity.xtarget_manifold_engines_and_crossings import (
    run_cross_target_manifold_engines_crossings_parity,
)
from parity.xtarget_metrics_spatial import run_cross_target_metrics_spatial_parity
from parity.xtarget_nd_solvers_and_metrics import run_cross_target_nd_solvers_metrics_parity
from parity.xtarget_neighborhood_knn import run_cross_target_neighborhood_knn_parity
from parity.xtarget_overlap_and_cluster_metrics import (
    run_cross_target_overlap_cluster_metrics_parity,
)
from parity.xtarget_overlap_constraints_layering import (
    run_cross_target_overlap_layering_parity,
)
from parity.xtarget_rn_tsnet_declutter_dijkstra import (
    run_cross_target_rn_tsnet_declutter_dijkstra_parity,
)
from parity.xtarget_solvers_backbone_poisson import (
    run_cross_target_solvers_backbone_poisson_parity,
)
from parity.xtarget_spatial_and_quality_metrics import (
    run_cross_target_spatial_quality_parity,
)


def run_csr_layout_python_suite(eg: Any) -> list[dict[str, Any]]:
    """Run all Python-side parity and adversarial checks for CsrGraph/Layout2d APIs."""
    results: list[dict[str, Any]] = []
    results.extend(run_core_csr_layout_checks(eg))
    results.extend(run_geometry_metrics_spatial_checks(eg))
    if hasattr(eg, "select_maxmin_sp_pivots"):
        results.extend(run_linalg_neighborhood_python_checks(eg))
    if hasattr(eg, "TsNetSolver"):
        results.extend(run_clustering_solvers_python_checks(eg))
    if hasattr(eg, "TFdpSolver"):
        results.extend(run_force_overlap_layering_python_checks(eg))
    if hasattr(eg, "cluster_area_faithfulness"):
        results.extend(run_gcaf_python_checks(eg))
    if hasattr(eg, "leiden_communities"):
        results.extend(run_community_baselines_python_checks(eg))
    if hasattr(eg, "hde"):
        results.extend(run_embedding_classic_solvers_python_checks(eg))

    assert not hasattr(eg, "InfoMap"), "InfoMap must not be exposed on egraph"
    assert "InfoMap" not in getattr(eg, "__all__", []), "InfoMap must not be in egraph.__all__"
    return results


def run_all_suites(
    wasm_pkg: pathlib.Path | None = None,
    wasm_only: bool = False,
) -> list[dict[str, Any]]:
    """Execute Python, WASM Node.js ESM, and cross-target Python <-> WASM parity suites."""
    results: list[dict[str, Any]] = []
    eg_mod: Any = None
    if not wasm_only:
        import egraph as eg

        eg_mod = eg
        if hasattr(eg, "CsrGraph") and hasattr(eg, "Layout2d"):
            results.extend(run_csr_layout_python_suite(eg))

    if wasm_pkg is not None:
        results.extend(run_wasm_node_parity(wasm_pkg))
        if eg_mod is not None:
            results.extend(run_cross_target_metrics_spatial_parity(eg_mod, wasm_pkg))
            results.extend(run_cross_target_linalg_spectral_parity(eg_mod, wasm_pkg))
            results.extend(run_cross_target_neighborhood_knn_parity(eg_mod, wasm_pkg))
            results.extend(run_cross_target_clustering_manifold_parity(eg_mod, wasm_pkg))
            results.extend(run_cross_target_core_solvers_parity(eg_mod, wasm_pkg))
            results.extend(run_cross_target_force_multilevel_parity(eg_mod, wasm_pkg))
            results.extend(run_cross_target_overlap_layering_parity(eg_mod, wasm_pkg))
            results.extend(run_cross_target_core_dataset_manifold_parity(eg_mod, wasm_pkg))
            results.extend(run_cross_target_spatial_quality_parity(eg_mod, wasm_pkg))
            results.extend(run_cross_target_gcaf_parity(eg_mod, wasm_pkg))
            results.extend(run_cross_target_embeddings_knn_probs_parity(eg_mod, wasm_pkg))
            results.extend(run_cross_target_solvers_backbone_poisson_parity(eg_mod, wasm_pkg))
            results.extend(run_cross_target_community_backbone_parity(eg_mod, wasm_pkg))
            results.extend(run_cross_target_overlap_cluster_metrics_parity(eg_mod, wasm_pkg))
            results.extend(run_cross_target_hde_sgd_presets_parity(eg_mod, wasm_pkg))
            results.extend(run_cross_target_classic_overlap_baselines_parity(eg_mod, wasm_pkg))
            results.extend(run_cross_target_nd_solvers_metrics_parity(eg_mod, wasm_pkg))
            results.extend(run_cross_target_manifold_engines_crossings_parity(eg_mod, wasm_pkg))
            results.extend(run_cross_target_rn_tsnet_declutter_dijkstra_parity(eg_mod, wasm_pkg))
            results.extend(run_cross_target_layout3d_generators_parity(eg_mod, wasm_pkg))
            results.extend(run_cross_target_error_contracts_parity(eg_mod, wasm_pkg))

    return results
