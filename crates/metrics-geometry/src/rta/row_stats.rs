use egraph_core::{CoreResult, CsrGraph, Layout2d, LayoutNd, ManifoldLayout2d, Scalar};
use egraph_linalg::BfsWorkspace;

pub(super) trait RtaCoordSpace {
    fn num_nodes(&self) -> usize;
    fn is_collapsed(&self) -> bool;
    fn dist_sq(&self, a: usize, b: usize) -> f64;
}

impl<S: Scalar> RtaCoordSpace for Layout2d<S> {
    #[inline]
    fn num_nodes(&self) -> usize {
        self.len()
    }

    #[inline]
    fn is_collapsed(&self) -> bool {
        is_layout_collapsed(self)
    }

    #[inline]
    fn dist_sq(&self, a: usize, b: usize) -> f64 {
        let dx = self.x()[b].to_f64() - self.x()[a].to_f64();
        let dy = self.y()[b].to_f64() - self.y()[a].to_f64();
        dx * dx + dy * dy
    }
}

impl<S: Scalar> RtaCoordSpace for LayoutNd<S> {
    #[inline]
    fn num_nodes(&self) -> usize {
        LayoutNd::num_nodes(self)
    }

    #[inline]
    fn is_collapsed(&self) -> bool {
        is_layout_collapsed_nd(self)
    }

    #[inline]
    fn dist_sq(&self, a: usize, b: usize) -> f64 {
        let dims = self.dims();
        let coords = self.coords();
        let mut sum = 0.0_f64;
        for d in 0..dims {
            let diff = coords[b * dims + d].to_f64() - coords[a * dims + d].to_f64();
            sum += diff * diff;
        }
        sum
    }
}

impl<S: Scalar> RtaCoordSpace for ManifoldLayout2d<S> {
    #[inline]
    fn num_nodes(&self) -> usize {
        self.len()
    }

    #[inline]
    fn is_collapsed(&self) -> bool {
        is_manifold_collapsed(self)
    }

    #[inline]
    fn dist_sq(&self, a: usize, b: usize) -> f64 {
        let d = self
            .geodesic_distance(a, b)
            .map(|v| v.to_f64())
            .unwrap_or(0.0);
        d * d
    }
}

#[inline]
pub(super) fn is_manifold_collapsed<S: Scalar>(layout: &ManifoldLayout2d<S>) -> bool {
    let n = layout.len();
    if n <= 1 {
        return true;
    }
    for i in 1..n {
        if let Ok(d) = layout.geodesic_distance(0, i)
            && d.to_f64() > 1e-15
        {
            return false;
        }
    }
    true
}

#[inline]
pub(super) fn is_layout_collapsed<S: Scalar>(layout: &Layout2d<S>) -> bool {
    let n = layout.len();
    if n == 0 {
        return true;
    }
    let xs = layout.x();
    let ys = layout.y();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for i in 0..n {
        let x = xs[i].to_f64();
        let y = ys[i].to_f64();
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    (max_x - min_x) <= 1e-15 && (max_y - min_y) <= 1e-15
}

#[inline]
pub(super) fn is_layout_collapsed_nd<S: Scalar>(layout: &LayoutNd<S>) -> bool {
    let n = layout.num_nodes();
    if n == 0 {
        return true;
    }
    let dims = layout.dims();
    let coords = layout.coords();
    for d in 0..dims {
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for i in 0..n {
            let v = coords[i * dims + d].to_f64();
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        if (max_v - min_v) > 1e-15 {
            return false;
        }
    }
    true
}

#[inline]
fn use_counting_sort(n: usize, max_d: u32) -> bool {
    (max_d as usize) <= n.max(1024).saturating_mul(4)
}

pub(super) struct RowScratch {
    dist_counts: Vec<u128>,
    dist_offsets: Vec<usize>,
    cursor: Vec<usize>,
    pub(super) sorted_nodes: Vec<u32>,
    pub(super) sorted_dists: Vec<u32>,
    pub(super) sorted_pairs: Vec<(u32, u32)>,
}

impl RowScratch {
    pub(super) fn new(n: usize) -> Self {
        Self {
            dist_counts: Vec::new(),
            dist_offsets: Vec::new(),
            cursor: Vec::new(),
            sorted_nodes: Vec::with_capacity(n),
            sorted_dists: Vec::with_capacity(n),
            sorted_pairs: Vec::new(),
        }
    }
}

pub(super) fn scan_row_octaves(
    row: &[u32],
    scratch: &mut RowScratch,
) -> ([u128; 32], [u128; 32], isize) {
    let mut oct_counts = [0_u128; 32];
    let mut same_pairs = [0_u128; 32];

    let mut max_d = 0_u32;
    for &d in row {
        if (1..u32::MAX).contains(&d) && d > max_d {
            max_d = d;
        }
    }
    if max_d == 0 {
        return (oct_counts, same_pairs, -1);
    }

    let mut sum_sq = [0_u128; 32];
    let mut max_oct = 0_isize;

    if use_counting_sort(row.len(), max_d) {
        let max_d_usize = max_d as usize;
        scratch.dist_counts.clear();
        scratch.dist_counts.resize(max_d_usize + 1, 0);
        for &d in row {
            if (1..u32::MAX).contains(&d) {
                scratch.dist_counts[d as usize] += 1;
            }
        }

        for (d, &cnt) in scratch
            .dist_counts
            .iter()
            .enumerate()
            .take(max_d_usize + 1)
            .skip(1)
        {
            if cnt > 0 {
                let b = (31 - (d as u32).leading_zeros()) as usize;
                if (b as isize) > max_oct {
                    max_oct = b as isize;
                }
                oct_counts[b] += cnt;
                sum_sq[b] += cnt * cnt;
            }
        }
    } else {
        // Fallback for large sparse/non-hop integer distances: O(N log N) sort in O(N) RAM.
        scratch.sorted_dists.clear();
        for &d in row {
            if (1..u32::MAX).contains(&d) {
                scratch.sorted_dists.push(d);
            }
        }
        scratch.sorted_dists.sort_unstable();
        let mut idx = 0usize;
        let len = scratch.sorted_dists.len();
        while idx < len {
            let d = scratch.sorted_dists[idx];
            let start = idx;
            while idx < len && scratch.sorted_dists[idx] == d {
                idx += 1;
            }
            let cnt = (idx - start) as u128;
            let b = (31 - d.leading_zeros()) as usize;
            if (b as isize) > max_oct {
                max_oct = b as isize;
            }
            oct_counts[b] += cnt;
            sum_sq[b] += cnt * cnt;
        }
    }

    for b in 0..=(max_oct as usize) {
        let cb = oct_counts[b];
        if cb >= 2 {
            same_pairs[b] = (cb * cb - sum_sq[b]) / 2;
        }
    }

    (oct_counts, same_pairs, max_oct)
}

pub(super) fn counting_sort_row(row: &[u32], scratch: &mut RowScratch) -> [usize; 33] {
    let mut oct_offsets = [0usize; 33];
    let mut max_d = 0_u32;
    let mut n_valid = 0usize;
    for &d in row {
        if (1..u32::MAX).contains(&d) {
            n_valid += 1;
            if d > max_d {
                max_d = d;
            }
        }
    }
    scratch.sorted_nodes.clear();
    scratch.sorted_dists.clear();
    if n_valid == 0 {
        return oct_offsets;
    }

    if use_counting_sort(row.len(), max_d) {
        let max_d_usize = max_d as usize;
        scratch.dist_counts.clear();
        scratch.dist_counts.resize(max_d_usize + 1, 0);
        for &d in row {
            if (1..u32::MAX).contains(&d) {
                scratch.dist_counts[d as usize] += 1;
            }
        }

        scratch.dist_offsets.clear();
        scratch.dist_offsets.resize(max_d_usize + 2, 0);
        let mut oct_counts = [0usize; 32];
        for d in 1..=max_d_usize {
            let cnt = scratch.dist_counts[d] as usize;
            scratch.dist_offsets[d + 1] = scratch.dist_offsets[d] + cnt;
            if cnt > 0 {
                let b = (31 - (d as u32).leading_zeros()) as usize;
                oct_counts[b] += cnt;
            }
        }
        for b in 0..32 {
            oct_offsets[b + 1] = oct_offsets[b] + oct_counts[b];
        }

        scratch.cursor.clear();
        scratch
            .cursor
            .extend_from_slice(&scratch.dist_offsets[..=max_d_usize]);
        scratch.sorted_nodes.resize(n_valid, 0);
        scratch.sorted_dists.resize(n_valid, 0);

        for (u, &d) in row.iter().enumerate() {
            if (1..u32::MAX).contains(&d) {
                let di = d as usize;
                let pos = scratch.cursor[di];
                scratch.sorted_nodes[pos] = u as u32;
                scratch.sorted_dists[pos] = d;
                scratch.cursor[di] = pos + 1;
            }
        }
    } else {
        // Fallback for large sparse/non-hop integer distances: sort (dist, node) in O(N) RAM.
        scratch.sorted_pairs.clear();
        scratch.sorted_pairs.reserve(n_valid);
        for (u, &d) in row.iter().enumerate() {
            if (1..u32::MAX).contains(&d) {
                scratch.sorted_pairs.push((d, u as u32));
            }
        }
        scratch.sorted_pairs.sort_unstable();
        scratch.sorted_nodes.reserve(n_valid);
        scratch.sorted_dists.reserve(n_valid);
        let mut oct_counts = [0usize; 32];
        for &(d, u) in &scratch.sorted_pairs {
            scratch.sorted_dists.push(d);
            scratch.sorted_nodes.push(u);
            let b = (31 - d.leading_zeros()) as usize;
            oct_counts[b] += 1;
        }
        for b in 0..32 {
            oct_offsets[b + 1] = oct_offsets[b] + oct_counts[b];
        }
    }

    oct_offsets
}

pub(super) enum RowProvider<'a, W> {
    Graph {
        graph: &'a CsrGraph<W>,
        ws: BfsWorkspace,
        buf: Vec<u32>,
    },
    Table {
        dists: &'a [u32],
        n: usize,
    },
}

impl<'a, W> RowProvider<'a, W> {
    pub(super) fn row(&mut self, landmark_idx: usize, landmark_id: u32) -> CoreResult<&[u32]> {
        match self {
            RowProvider::Graph { graph, ws, buf } => {
                ws.bfs_from_u32(graph, landmark_id as usize, buf)?;
                Ok(buf.as_slice())
            }
            RowProvider::Table { dists, n } => {
                let start = landmark_idx * *n;
                Ok(&dists[start..start + *n])
            }
        }
    }
}
