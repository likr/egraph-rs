use egraph_core::{
    CoreResult, CsrGraph, DeterministicRng, EuclideanLayout, EuclideanStorageMut, Layout2d,
    Layout3d, LayoutGenerator, LayoutNd, Scalar,
};
use egraph_linalg_mds::{Hde, HdeConfig, PivotMds, PivotMdsConfig};

use super::graph_prep::is_layout_collapsed_euclidean;
use super::solver::MaxentCoreState;

fn build_init_graph(core: &MaxentCoreState) -> CoreResult<CsrGraph<f64>> {
    let n = core.num_nodes;
    let edge_pairs: Vec<(u32, u32)> = core.edges.iter().map(|e| (e.u, e.v)).collect();
    if core.all_unit_edges {
        CsrGraph::<f64>::from_unweighted_edges(n, &edge_pairs, false, false)
    } else {
        let edge_dists: Vec<f64> = core.edges.iter().map(|e| e.d).collect();
        CsrGraph::<f64>::from_edges(n, &edge_pairs, Some(&edge_dists), false, false)
    }
}

pub(super) fn initialize_collapsed_euclidean<L: EuclideanLayout>(
    core: &MaxentCoreState,
    layout: &mut L,
) -> CoreResult<()> {
    let n = core.num_nodes;
    let init_graph = build_init_graph(core)?;

    match layout.as_euclidean_storage_mut() {
        EuclideanStorageMut::Soa2d { x, y } => {
            let pmds_cfg = PivotMdsConfig::new().with_seed(core.config.seed);
            let mut generator =
                PivotMds::<Layout2d<L::Scalar>, f64>::from_config(&init_graph, pmds_cfg);
            if let Ok(pmds_layout) = generator.generate()
                && !is_layout_collapsed_euclidean(&pmds_layout)
            {
                x.copy_from_slice(pmds_layout.x());
                y.copy_from_slice(pmds_layout.y());
                return Ok(());
            }

            let mut rng = DeterministicRng::from_seed_u64(core.config.seed);
            for i in 0..n {
                x[i] = L::Scalar::from_f64(rng.next_f64() - 0.5);
                y[i] = L::Scalar::from_f64(rng.next_f64() - 0.5);
            }
        }
        EuclideanStorageMut::Soa3d { x, y, z } => {
            let hde_cfg = HdeConfig::new().with_seed(core.config.seed);
            let mut generator = Hde::<Layout3d<L::Scalar>, f64>::from_config(&init_graph, hde_cfg);
            if let Ok(hde_layout) = generator.generate()
                && !is_layout_collapsed_euclidean(&hde_layout)
            {
                x.copy_from_slice(hde_layout.x());
                y.copy_from_slice(hde_layout.y());
                z.copy_from_slice(hde_layout.z());
                return Ok(());
            }

            let mut rng = DeterministicRng::from_seed_u64(core.config.seed);
            for i in 0..n {
                x[i] = L::Scalar::from_f64(rng.next_f64() - 0.5);
                y[i] = L::Scalar::from_f64(rng.next_f64() - 0.5);
                z[i] = L::Scalar::from_f64(rng.next_f64() - 0.5);
            }
        }
        EuclideanStorageMut::FlatNd { coords, dims, .. } => {
            let hde_cfg = HdeConfig::new()
                .with_output_dims(Some(dims))
                .with_seed(core.config.seed);
            let mut generator = Hde::<LayoutNd<L::Scalar>, f64>::from_config(&init_graph, hde_cfg);
            if let Ok(hde_layout) = generator.generate()
                && !is_layout_collapsed_euclidean(&hde_layout)
            {
                coords.copy_from_slice(hde_layout.coords());
                return Ok(());
            }

            let mut rng = DeterministicRng::from_seed_u64(core.config.seed);
            for v in coords.iter_mut() {
                *v = L::Scalar::from_f64(rng.next_f64() - 0.5);
            }
        }
    }

    Ok(())
}
