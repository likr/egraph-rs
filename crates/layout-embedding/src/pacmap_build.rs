use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Scalar};
use egraph_linalg::{BfsWorkspace, select_maxmin_sp_pivots};
use egraph_neighborhood::DogVaradhanLandmarks;

use crate::config_pacmap_snap::PacmapGraphConfig;
use crate::pacmap_pairs::{LandmarkTable, PacmapTierPairs, collect_pacmap_tier_pairs};
use crate::sampling_utils::{
    MAX_PACKED_NODES, build_symmetric_csr_from_sorted_pairs, extract_canonical_undirected_edges,
    validate_graph_weights,
};

pub(crate) enum PacmapExternalLandmarks<'a> {
    Dog(&'a DogVaradhanLandmarks),
    Distances {
        data: &'a [f64],
        num_landmarks: usize,
    },
}

pub(crate) struct BuiltPacmapData {
    pub(crate) num_nodes: usize,
    pub(crate) nb_pairs: Vec<(u32, u32)>,
    pub(crate) nb_weights: Vec<f64>,
    pub(crate) mn_pairs: Vec<(u32, u32)>,
    pub(crate) fp_pairs: Vec<(u32, u32)>,
}

fn validate_pacmap_external_landmarks(
    n: usize,
    ext_landmarks: Option<&PacmapExternalLandmarks<'_>>,
) -> CoreResult<()> {
    let Some(ext) = ext_landmarks else {
        return Ok(());
    };
    match ext {
        PacmapExternalLandmarks::Dog(landmarks) => {
            if landmarks.num_nodes() != n {
                return Err(CoreError::dimension_mismatch(
                    n,
                    landmarks.num_nodes(),
                    "PacmapGraphSolver::with_landmarks::landmarks",
                ));
            }
            let k_lm = landmarks.num_landmarks();
            if n > 0 && k_lm == 0 {
                return Err(CoreError::invalid_input(
                    "landmarks must have at least 1 landmark when num_nodes > 0",
                ));
            }
            let expected = n
                .checked_mul(k_lm)
                .ok_or_else(|| CoreError::invalid_input("landmark table size overflow"))?;
            if landmarks.d_dog().len() != expected {
                return Err(CoreError::dimension_mismatch(
                    expected,
                    landmarks.d_dog().len(),
                    "DogVaradhanLandmarks::d_dog",
                ));
            }
            if landmarks.d_dog().iter().any(|v| !v.is_finite()) {
                return Err(CoreError::non_finite("DogVaradhanLandmarks::d_dog"));
            }
            if let Some(dc) = landmarks.d_coarse() {
                if dc.len() != expected {
                    return Err(CoreError::dimension_mismatch(
                        expected,
                        dc.len(),
                        "DogVaradhanLandmarks::d_coarse",
                    ));
                }
                if dc.iter().any(|v| !v.is_finite()) {
                    return Err(CoreError::non_finite("DogVaradhanLandmarks::d_coarse"));
                }
            }
            for &lm in landmarks.landmarks() {
                if (lm as usize) >= n {
                    return Err(CoreError::index_out_of_bounds(
                        lm as usize,
                        n,
                        "DogVaradhanLandmarks::landmarks",
                    ));
                }
            }
        }
        PacmapExternalLandmarks::Distances {
            data,
            num_landmarks,
        } => {
            let k_lm = *num_landmarks;
            if n > 0 && k_lm == 0 {
                return Err(CoreError::invalid_input(
                    "num_landmarks must be >= 1 when num_nodes > 0",
                ));
            }
            let expected = n
                .checked_mul(k_lm)
                .ok_or_else(|| CoreError::invalid_input("landmark table size overflow"))?;
            if data.len() != expected {
                return Err(CoreError::dimension_mismatch(
                    expected,
                    data.len(),
                    "PacmapGraphSolver::with_landmark_distances::landmark_distances",
                ));
            }
            for &v in *data {
                if !v.is_finite() {
                    return Err(CoreError::non_finite(
                        "PacmapGraphSolver::with_landmark_distances::landmark_distances",
                    ));
                }
                if v < 0.0 {
                    return Err(CoreError::invalid_input(format!(
                        "landmark_distances must be >= 0.0, got {v}"
                    )));
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn build_pacmap_data<W: Scalar>(
    graph: &CsrGraph<W>,
    config: &PacmapGraphConfig,
    ext_landmarks: Option<PacmapExternalLandmarks<'_>>,
) -> CoreResult<BuiltPacmapData> {
    config.validate()?;
    validate_graph_weights(graph, "PacmapGraphSolver::new")?;

    let n = graph.num_nodes();
    if n > MAX_PACKED_NODES {
        return Err(CoreError::invalid_input(format!(
            "num_nodes ({n}) exceeds packed u32 index limit {MAX_PACKED_NODES}"
        )));
    }
    validate_pacmap_external_landmarks(n, ext_landmarks.as_ref())?;

    if n <= 1 {
        return Ok(BuiltPacmapData {
            num_nodes: n,
            nb_pairs: Vec::new(),
            nb_weights: Vec::new(),
            mn_pairs: Vec::new(),
            fp_pairs: Vec::new(),
        });
    }

    let canonical_edges = extract_canonical_undirected_edges(graph);
    let n_canon_edges = canonical_edges.len();
    let mut u1 = Vec::with_capacity(n_canon_edges);
    let mut v1 = Vec::with_capacity(n_canon_edges);
    for &(u, v, _) in &canonical_edges {
        u1.push(u);
        v1.push(v);
    }
    let (adj_indptr, adj_indices) = build_symmetric_csr_from_sorted_pairs(n, &u1, &v1);
    let sym_graph = CsrGraph::<f64>::from_unweighted_csr(&adj_indptr, &adj_indices, false, false)?;

    let weighted_rows: Option<Vec<Vec<(f64, u32)>>> = if graph.is_weighted() {
        let mut rows = vec![Vec::new(); n];
        for &(u, v, w) in &canonical_edges {
            rows[u as usize].push((w, v));
            rows[v as usize].push((w, u));
        }
        for row in &mut rows {
            row.sort_unstable_by(|a, b| {
                a.0.total_cmp(&b.0)
                    .then_with(|| {
                        sym_graph
                            .degree(a.1 as usize)
                            .cmp(&sym_graph.degree(b.1 as usize))
                    })
                    .then_with(|| a.1.cmp(&b.1))
            });
        }
        Some(rows)
    } else {
        None
    };
    drop(canonical_edges);
    drop(u1);
    drop(v1);

    let n_nb = config.n_neighbors.min(n - 1);
    let n_mn = ((config.mn_ratio * (config.n_neighbors as f64)).floor() as usize).min(n - 1);
    let n_fp = ((config.fp_ratio * (config.n_neighbors as f64)).floor() as usize).min(n - 1);

    let lm_table = if n_mn == 0 {
        LandmarkTable::None
    } else {
        match ext_landmarks {
            Some(PacmapExternalLandmarks::Dog(landmarks)) => LandmarkTable::RowMajor {
                data: landmarks.d_dog(),
                num_landmarks: landmarks.num_landmarks(),
            },
            Some(PacmapExternalLandmarks::Distances {
                data,
                num_landmarks,
            }) => LandmarkTable::RowMajor {
                data,
                num_landmarks,
            },
            None => {
                let k_lm = config.num_landmarks.min(n);
                let landmarks = select_maxmin_sp_pivots(&sym_graph, k_lm, config.seed, 2)?;
                let num_lm = landmarks.len();
                let mut lm_dists = vec![f64::INFINITY; num_lm * n];
                let mut bfs = BfsWorkspace::new(n);
                for (idx, &lm) in landmarks.iter().enumerate() {
                    bfs.bfs_from(
                        &sym_graph,
                        lm as usize,
                        &mut lm_dists[idx * n..(idx + 1) * n],
                    )?;
                }
                LandmarkTable::ColMajor {
                    data: lm_dists,
                    num_landmarks: num_lm,
                }
            }
        }
    };

    let mut rng = DeterministicRng::from_seed_u64(config.seed);
    let PacmapTierPairs {
        mut nb_pairs,
        mn_pairs,
        fp_pairs,
    } = collect_pacmap_tier_pairs(
        &sym_graph,
        &adj_indptr,
        weighted_rows.as_deref(),
        &lm_table,
        n,
        n_canon_edges,
        n_nb,
        n_mn,
        n_fp,
        config.keep_all_1hop,
        config.max_1hop_per_node,
        &mut rng,
    );

    nb_pairs.sort_unstable();
    nb_pairs.dedup();
    let nb_weights: Vec<f64> = if config.degree_weight_beta > 0.0 {
        let beta = config.degree_weight_beta;
        let deg_scale: Vec<f64> = (0..n)
            .map(|u| (sym_graph.degree(u).max(1) as f64).powf(-beta))
            .collect();
        nb_pairs
            .iter()
            .map(|&(u, v)| deg_scale[u as usize] * deg_scale[v as usize])
            .collect()
    } else {
        Vec::new()
    };

    Ok(BuiltPacmapData {
        num_nodes: n,
        nb_pairs,
        nb_weights,
        mn_pairs,
        fp_pairs,
    })
}
