//! Cooperative layout solver, density refiner, overlap removal, and edge bundling bindings.

mod backbone;
mod cluster_refine;
mod coarse_backbone;
mod distance_extract;
mod edge_bundling;
mod force_baselines;
mod force_classic;
mod force_multilevel;
mod group_in_a_box;
mod gumap;
mod gumap_builders;
mod ib_umap;
mod ib_umap_builders;
mod layout_dispatch;
mod overlap_prism_pfs;
mod overlap_tree_forbid;
mod screened_poisson_field;
mod screened_poisson_refiner;
mod sfc;
mod sgd;
mod sgd_barrier;
mod sgd_baselines;
mod sgd_core;
mod sgd_dispatch;
mod sgd_pacmap;
mod spatial_refiner;
mod stress_kamada;
mod stress_multilevel;
mod t_fdp;
mod tsnet;
mod tsnet_builders;
mod tsnet_config;
mod yifan_hu;

#[cfg(test)]
mod tests;

use pyo3::prelude::*;

use edge_bundling::PyFdebSolver;
use force_baselines::{PyGemSolver, PySelLayoutSolver};
use force_classic::{PyFruchtermanReingoldSolver, PyLinLogSolver};
use force_multilevel::PyForceAtlas2Solver;
use gumap::PyGuMapSolver;
use ib_umap::PyIbUmapSolver;
use overlap_prism_pfs::{PyPfsPrimeOverlapRemoval, PyPrismOverlapRemoval};
use overlap_tree_forbid::{PyForbidOverlapRemoval, PyGTreeOverlapRemoval};
use screened_poisson_field::PyScreenedPoissonField2d;
use screened_poisson_refiner::PyScreenedPoissonRefiner;
use sfc::PySfcLayoutSolver;
use sgd::PySgdSolver;
use spatial_refiner::PySpatialIntruderRefiner;
use stress_kamada::{PyKamadaKawaiSolver, PyStressMajorizationSolver};
use stress_multilevel::{PyGripSolver, PyMaxentStressSolver};
use t_fdp::PyTFdpSolver;
use tsnet::PyTsNetSolver;
use yifan_hu::PyYifanHuSolver;

/// Register cooperative layout solver, overlap removal, and edge bundling bindings on `egraph._core`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTsNetSolver>()?;
    tsnet_config::register(m)?;
    m.add_class::<PySgdSolver>()?;
    m.add_class::<PyGuMapSolver>()?;
    m.add_class::<PySpatialIntruderRefiner>()?;
    m.add_class::<PyStressMajorizationSolver>()?;
    m.add_class::<PyKamadaKawaiSolver>()?;
    m.add_class::<PyTFdpSolver>()?;
    m.add_class::<PyIbUmapSolver>()?;
    m.add_class::<PyForceAtlas2Solver>()?;
    m.add_class::<PyYifanHuSolver>()?;
    m.add_class::<PyPrismOverlapRemoval>()?;
    m.add_class::<PyFdebSolver>()?;
    m.add_class::<PyScreenedPoissonField2d>()?;
    m.add_class::<PyScreenedPoissonRefiner>()?;
    m.add_class::<PyLinLogSolver>()?;
    m.add_class::<PyFruchtermanReingoldSolver>()?;
    backbone::register(m)?;
    cluster_refine::register(m)?;
    group_in_a_box::register(m)?;
    m.add_class::<PyGTreeOverlapRemoval>()?;
    m.add_class::<PyForbidOverlapRemoval>()?;
    m.add_class::<PyMaxentStressSolver>()?;
    m.add_class::<PyGripSolver>()?;
    m.add_class::<PyGemSolver>()?;
    m.add_class::<PySelLayoutSolver>()?;
    m.add_class::<PyPfsPrimeOverlapRemoval>()?;
    m.add_class::<PySfcLayoutSolver>()?;
    coarse_backbone::register(m)?;
    Ok(())
}
