use egraph_core::{
    CoreResult, CsrGraph, DeterministicRng, Layout2d, Layout3d, LayoutGenerator, LayoutNd, Scalar,
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

pub(super) fn initialize_collapsed_2d<S: Scalar>(
    core: &MaxentCoreState,
    layout: &mut Layout2d<S>,
) -> CoreResult<()> {
    let n = core.num_nodes;
    let init_graph = build_init_graph(core)?;
    let pmds_cfg = PivotMdsConfig::new().with_seed(core.config.seed);
    let mut generator = PivotMds::<Layout2d<S>, f64>::from_config(&init_graph, pmds_cfg);
    if let Ok(pmds_layout) = generator.generate()
        && !is_layout_collapsed_euclidean(&pmds_layout)
    {
        let (dst_x, dst_y) = layout.xy_mut();
        dst_x.copy_from_slice(pmds_layout.x());
        dst_y.copy_from_slice(pmds_layout.y());
        return Ok(());
    }

    let mut rng = DeterministicRng::from_seed_u64(core.config.seed);
    let (dst_x, dst_y) = layout.xy_mut();
    for i in 0..n {
        dst_x[i] = S::from_f64(rng.next_f64() - 0.5);
        dst_y[i] = S::from_f64(rng.next_f64() - 0.5);
    }
    Ok(())
}

pub(super) fn initialize_collapsed_3d<S: Scalar>(
    core: &MaxentCoreState,
    layout: &mut Layout3d<S>,
) -> CoreResult<()> {
    let n = core.num_nodes;
    let init_graph = build_init_graph(core)?;
    let hde_cfg = HdeConfig::new().with_seed(core.config.seed);
    let mut generator = Hde::<Layout3d<S>, f64>::from_config(&init_graph, hde_cfg);
    if let Ok(hde_layout) = generator.generate()
        && !is_layout_collapsed_euclidean(&hde_layout)
    {
        let (dst_x, dst_y, dst_z) = layout.xyz_mut();
        dst_x.copy_from_slice(hde_layout.x());
        dst_y.copy_from_slice(hde_layout.y());
        dst_z.copy_from_slice(hde_layout.z());
        return Ok(());
    }

    let mut rng = DeterministicRng::from_seed_u64(core.config.seed);
    let (dst_x, dst_y, dst_z) = layout.xyz_mut();
    for i in 0..n {
        dst_x[i] = S::from_f64(rng.next_f64() - 0.5);
        dst_y[i] = S::from_f64(rng.next_f64() - 0.5);
        dst_z[i] = S::from_f64(rng.next_f64() - 0.5);
    }
    Ok(())
}

pub(super) fn initialize_collapsed_nd<S: Scalar>(
    core: &MaxentCoreState,
    layout: &mut LayoutNd<S>,
) -> CoreResult<()> {
    let d = layout.dims();
    let init_graph = build_init_graph(core)?;
    let hde_cfg = HdeConfig::new()
        .with_output_dims(Some(d))
        .with_seed(core.config.seed);
    let mut generator = Hde::<LayoutNd<S>, f64>::from_config(&init_graph, hde_cfg);
    if let Ok(hde_layout) = generator.generate()
        && !is_layout_collapsed_euclidean(&hde_layout)
    {
        layout.coords_mut().copy_from_slice(hde_layout.coords());
        return Ok(());
    }

    let mut rng = DeterministicRng::from_seed_u64(core.config.seed);
    for v in layout.coords_mut().iter_mut() {
        *v = S::from_f64(rng.next_f64() - 0.5);
    }
    Ok(())
}
