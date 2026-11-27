//! 2-stage coarse `StaticKdTree` + full-$D$ Euclidean reranking $k$-NN builder and shared
//! coordinate validation/selection utilities.

use std::cmp::Ordering;

use egraph_core::{CoreError, CoreResult};
use egraph_spatial_trees::{KdQueryScratch, NearestNeighborIndex, StaticKdTree};

use crate::knn::KnnGraph;

#[inline]
pub(super) fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    let mut sum_sq = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        sum_sq += diff * diff;
    }
    sum_sq.sqrt()
}

pub(super) fn validate_and_center_coords(
    coords_row_major: &[f64],
    n_nodes: usize,
    dim: usize,
    context: &'static str,
) -> CoreResult<Vec<f64>> {
    if u32::try_from(n_nodes).is_err() {
        return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
    }
    let expected = n_nodes
        .checked_mul(dim)
        .ok_or_else(|| CoreError::invalid_input("coordinate buffer size overflow"))?;
    if coords_row_major.len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            coords_row_major.len(),
            context,
        ));
    }
    if coords_row_major.iter().any(|v| !v.is_finite()) {
        return Err(CoreError::non_finite(context));
    }
    if n_nodes == 0 || dim == 0 {
        return Ok(Vec::new());
    }

    let mut means = vec![0.0_f64; dim];
    for i in 0..n_nodes {
        let row = &coords_row_major[i * dim..(i + 1) * dim];
        for d in 0..dim {
            means[d] += row[d];
        }
    }
    let inv_n = 1.0 / (n_nodes as f64);
    for m in &mut means {
        *m *= inv_n;
    }

    let mut centered = vec![0.0_f64; expected];
    for i in 0..n_nodes {
        let src = &coords_row_major[i * dim..(i + 1) * dim];
        let dst = &mut centered[i * dim..(i + 1) * dim];
        for d in 0..dim {
            dst[d] = src[d] - means[d];
        }
    }
    Ok(centered)
}

#[inline]
pub(super) fn select_top_k_pairs(scored: &mut Vec<(f64, u32)>, k: usize) {
    let cmp = |a: &(f64, u32), b: &(f64, u32)| -> Ordering {
        a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1))
    };
    if scored.len() > k && k > 0 {
        scored.select_nth_unstable_by(k - 1, cmp);
        scored.truncate(k);
    }
    scored.sort_unstable_by(cmp);
}

pub(super) enum CoarseKdTree {
    D1(StaticKdTree<1, f64>),
    D2(StaticKdTree<2, f64>),
    D3(StaticKdTree<3, f64>),
    D4(StaticKdTree<4, f64>),
    D5(StaticKdTree<5, f64>),
    D6(StaticKdTree<6, f64>),
    D7(StaticKdTree<7, f64>),
    D8(StaticKdTree<8, f64>),
    D9(StaticKdTree<9, f64>),
    D10(StaticKdTree<10, f64>),
    D11(StaticKdTree<11, f64>),
    D12(StaticKdTree<12, f64>),
}

impl CoarseKdTree {
    pub(super) fn build(
        centered: &[f64],
        n_nodes: usize,
        dim: usize,
        coarse_dim: usize,
    ) -> CoreResult<Self> {
        let cd = coarse_dim.min(dim).clamp(1, 12);
        match cd {
            1 => Ok(Self::D1(build_typed_tree::<1>(centered, n_nodes, dim)?)),
            2 => Ok(Self::D2(build_typed_tree::<2>(centered, n_nodes, dim)?)),
            3 => Ok(Self::D3(build_typed_tree::<3>(centered, n_nodes, dim)?)),
            4 => Ok(Self::D4(build_typed_tree::<4>(centered, n_nodes, dim)?)),
            5 => Ok(Self::D5(build_typed_tree::<5>(centered, n_nodes, dim)?)),
            6 => Ok(Self::D6(build_typed_tree::<6>(centered, n_nodes, dim)?)),
            7 => Ok(Self::D7(build_typed_tree::<7>(centered, n_nodes, dim)?)),
            8 => Ok(Self::D8(build_typed_tree::<8>(centered, n_nodes, dim)?)),
            9 => Ok(Self::D9(build_typed_tree::<9>(centered, n_nodes, dim)?)),
            10 => Ok(Self::D10(build_typed_tree::<10>(centered, n_nodes, dim)?)),
            11 => Ok(Self::D11(build_typed_tree::<11>(centered, n_nodes, dim)?)),
            _ => Ok(Self::D12(build_typed_tree::<12>(centered, n_nodes, dim)?)),
        }
    }

    #[inline]
    pub(super) fn as_index(&self) -> &dyn NearestNeighborIndex {
        match self {
            Self::D1(t) => t,
            Self::D2(t) => t,
            Self::D3(t) => t,
            Self::D4(t) => t,
            Self::D5(t) => t,
            Self::D6(t) => t,
            Self::D7(t) => t,
            Self::D8(t) => t,
            Self::D9(t) => t,
            Self::D10(t) => t,
            Self::D11(t) => t,
            Self::D12(t) => t,
        }
    }

    pub(super) fn k_nearest_excluding_self_with_scratch(
        &self,
        point_idx: usize,
        k: usize,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        self.as_index()
            .query_knn_excluding_self_with_scratch(point_idx, k, scratch, out)
    }
}

fn build_typed_tree<const D: usize>(
    centered: &[f64],
    n_nodes: usize,
    dim: usize,
) -> CoreResult<StaticKdTree<D, f64>> {
    let mut pts = Vec::with_capacity(n_nodes);
    for i in 0..n_nodes {
        let mut p = [0.0_f64; D];
        let row = &centered[i * dim..(i + 1) * dim];
        p.copy_from_slice(&row[..D]);
        pts.push(p);
    }
    StaticKdTree::<D, f64>::build(&pts)
}

/// Constructs an exact/2-stage reranked $k$-NN [`KnnGraph`] from row-major `(n_nodes, dim)`
/// coordinates using a coarse [`StaticKdTree`] over `1..=12` dimensions followed by exact
/// full-`dim` Euclidean reranking.
///
/// Automatically falls back to deterministic exact full-`dim` search when `cand_k >= n_nodes - 1`.
pub fn build_rerank_knn(
    coords_row_major: &[f64],
    n_nodes: usize,
    dim: usize,
    k: usize,
    coarse_dim: usize,
    rerank_factor: usize,
) -> CoreResult<KnnGraph> {
    if coarse_dim == 0 {
        return Err(CoreError::invalid_input("coarse_dim must be >= 1"));
    }
    if rerank_factor == 0 {
        return Err(CoreError::invalid_input("rerank_factor must be >= 1"));
    }
    let centered = validate_and_center_coords(coords_row_major, n_nodes, dim, "build_rerank_knn")?;
    if n_nodes <= 1 || k == 0 {
        return KnnGraph::empty(n_nodes);
    }

    let k_eff = k.min(n_nodes - 1);
    let mut indptr = Vec::with_capacity(n_nodes + 1);
    indptr.push(0u32);
    let mut indices = Vec::with_capacity(n_nodes * k_eff);
    let mut distances = Vec::with_capacity(n_nodes * k_eff);

    if dim == 0 {
        for i in 0..n_nodes {
            let mut added = 0usize;
            for j in 0..n_nodes {
                if j == i {
                    continue;
                }
                indices.push(j as u32);
                distances.push(0.0);
                added += 1;
                if added == k_eff {
                    break;
                }
            }
            let ptr = u32::try_from(indices.len())
                .map_err(|_| CoreError::invalid_csr("rerank knn nnz exceeds u32::MAX"))?;
            indptr.push(ptr);
        }
        return KnnGraph::new_validated(n_nodes, indptr, indices, distances);
    }

    let cand_k = rerank_factor
        .saturating_mul(k_eff)
        .max(k_eff)
        .min(n_nodes - 1);
    let mut scored: Vec<(f64, u32)> = Vec::with_capacity(cand_k);

    if cand_k >= n_nodes - 1 {
        for i in 0..n_nodes {
            scored.clear();
            let row_i = &centered[i * dim..(i + 1) * dim];
            for j in 0..n_nodes {
                if j == i {
                    continue;
                }
                let row_j = &centered[j * dim..(j + 1) * dim];
                scored.push((euclidean_dist(row_i, row_j), j as u32));
            }
            select_top_k_pairs(&mut scored, k_eff);
            for &(dist, idx) in &scored {
                indices.push(idx);
                distances.push(dist);
            }
            let ptr = u32::try_from(indices.len())
                .map_err(|_| CoreError::invalid_csr("rerank knn nnz exceeds u32::MAX"))?;
            indptr.push(ptr);
        }
    } else {
        let tree = CoarseKdTree::build(&centered, n_nodes, dim, coarse_dim)?;
        let mut kd_scratch = KdQueryScratch::with_capacity(cand_k);
        let mut cand_buf: Vec<(usize, f64)> = Vec::with_capacity(cand_k);
        for i in 0..n_nodes {
            tree.k_nearest_excluding_self_with_scratch(i, cand_k, &mut kd_scratch, &mut cand_buf)?;
            scored.clear();
            let row_i = &centered[i * dim..(i + 1) * dim];
            for &(j, _) in &cand_buf {
                let row_j = &centered[j * dim..(j + 1) * dim];
                scored.push((euclidean_dist(row_i, row_j), j as u32));
            }
            select_top_k_pairs(&mut scored, k_eff);
            for &(dist, idx) in &scored {
                indices.push(idx);
                distances.push(dist);
            }
            let ptr = u32::try_from(indices.len())
                .map_err(|_| CoreError::invalid_csr("rerank knn nnz exceeds u32::MAX"))?;
            indptr.push(ptr);
        }
    }

    KnnGraph::new_validated(n_nodes, indptr, indices, distances)
}
