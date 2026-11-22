//! Flat median-split static $D$-dimensional KD-tree (`StaticKdTree`).
//!
//! Stores permuted points in contiguous leaf slices with tight bounding boxes per internal
//! node, accumulating squared Euclidean distances in `f64` and breaking ties deterministically
//! by ascending original point index for exact parity with $O(N^2)$ brute-force search.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::marker::PhantomData;

use egraph_core::{CoreError, CoreResult, Layout2d, LayoutNd, Scalar};

const LEAF_SIZE: usize = 16;

#[derive(Debug, Clone, Copy)]
struct KdEntry<const D: usize> {
    coords: [f64; D],
    index: u32,
}

#[derive(Debug, Clone, Copy)]
struct KdNode<const D: usize> {
    min_bounds: [f64; D],
    max_bounds: [f64; D],
    start: usize,
    end: usize,
    split_dim: usize,
    split_val: f64,
    left: u32,
    right: u32,
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    dist_sq: f64,
    index: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist_sq.total_cmp(&other.dist_sq) == Ordering::Equal && self.index == other.index
    }
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist_sq
            .total_cmp(&other.dist_sq)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Flat static median-split KD-tree in $D$ dimensions.
///
/// Distance computations and bounding-box pruning always execute in `f64` regardless of
/// whether the input coordinates are `f32` or `f64`.
#[derive(Debug, Clone)]
pub struct StaticKdTree<const D: usize, S: Scalar = f64> {
    nodes: Vec<KdNode<D>>,
    entries: Vec<KdEntry<D>>,
    coords_by_id: Vec<[f64; D]>,
    _marker: PhantomData<S>,
}

impl<const D: usize, S: Scalar> StaticKdTree<D, S> {
    /// Builds a static KD-tree from a slice of $D$-dimensional points.
    ///
    /// Returns [`CoreError::InvalidInput`] when `D == 0` and [`CoreError::NonFiniteValue`]
    /// if any coordinate is `NaN` or infinite.
    pub fn build(points: &[[S; D]]) -> CoreResult<Self> {
        if D == 0 {
            return Err(CoreError::invalid_input(
                "StaticKdTree dimension D must be >= 1",
            ));
        }
        let n = points.len();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_input(
                "StaticKdTree point count exceeds u32::MAX",
            ));
        }

        let mut coords_by_id = Vec::with_capacity(n);
        let mut entries = Vec::with_capacity(n);
        for (i, pt) in points.iter().enumerate() {
            let mut c = [0.0_f64; D];
            for d in 0..D {
                if !pt[d].is_finite() {
                    return Err(CoreError::non_finite("StaticKdTree::build"));
                }
                c[d] = pt[d].to_f64();
            }
            coords_by_id.push(c);
            entries.push(KdEntry {
                coords: c,
                index: i as u32,
            });
        }

        let mut tree = Self {
            nodes: Vec::new(),
            entries,
            coords_by_id,
            _marker: PhantomData,
        };
        if n > 0 {
            let est_nodes = (2 * n / LEAF_SIZE).max(8);
            tree.nodes.reserve(est_nodes);
            tree.build_node(0, n);
        }
        Ok(tree)
    }

    /// Builds a $D$-dimensional KD-tree from a flat [`LayoutNd<S>`].
    pub fn from_layout_nd(layout: &LayoutNd<S>) -> CoreResult<Self> {
        if D == 0 {
            return Err(CoreError::invalid_input(
                "StaticKdTree dimension D must be >= 1",
            ));
        }
        if layout.dims() != D {
            return Err(CoreError::dimension_mismatch(
                D,
                layout.dims(),
                "StaticKdTree::from_layout_nd",
            ));
        }
        if !layout.is_all_finite() {
            return Err(CoreError::non_finite("StaticKdTree::from_layout_nd"));
        }
        let n = layout.num_nodes();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_input(
                "StaticKdTree point count exceeds u32::MAX",
            ));
        }

        let flat = layout.coords();
        let mut coords_by_id = Vec::with_capacity(n);
        let mut entries = Vec::with_capacity(n);
        for i in 0..n {
            let mut c = [0.0_f64; D];
            let base = i * D;
            for d in 0..D {
                c[d] = flat[base + d].to_f64();
            }
            coords_by_id.push(c);
            entries.push(KdEntry {
                coords: c,
                index: i as u32,
            });
        }

        let mut tree = Self {
            nodes: Vec::new(),
            entries,
            coords_by_id,
            _marker: PhantomData,
        };
        if n > 0 {
            tree.nodes.reserve((2 * n / LEAF_SIZE).max(8));
            tree.build_node(0, n);
        }
        Ok(tree)
    }

    /// Returns the number of points stored in the tree.
    #[inline]
    pub fn len(&self) -> usize {
        self.coords_by_id.len()
    }

    /// Returns `true` if the tree stores zero points.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.coords_by_id.is_empty()
    }

    /// Returns the `f64` coordinates of original point `point_idx`.
    pub fn point(&self, point_idx: usize) -> CoreResult<[f64; D]> {
        self.coords_by_id.get(point_idx).copied().ok_or_else(|| {
            CoreError::index_out_of_bounds(point_idx, self.len(), "StaticKdTree::point")
        })
    }

    /// Returns the `k` nearest points `(index, dist_sq)` to `query`, sorted by `(dist_sq, index)`.
    pub fn k_nearest(&self, query: &[S; D], k: usize) -> CoreResult<Vec<(usize, f64)>> {
        let mut out = Vec::new();
        self.k_nearest_into(query, k, &mut out)?;
        Ok(out)
    }

    /// Reuses `out` to return the `k` nearest points `(index, dist_sq)` to `query`.
    pub fn k_nearest_into(
        &self,
        query: &[S; D],
        k: usize,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let q = convert_query(query)?;
        out.clear();
        if k == 0 || self.nodes.is_empty() {
            return Ok(());
        }
        self.knn_search(&q, k, None, out);
        Ok(())
    }

    /// Returns the `k` nearest points `(index, dist_sq)` to point `point_idx`, excluding `point_idx` itself.
    pub fn k_nearest_excluding_self(
        &self,
        point_idx: usize,
        k: usize,
    ) -> CoreResult<Vec<(usize, f64)>> {
        let mut out = Vec::new();
        self.k_nearest_excluding_self_into(point_idx, k, &mut out)?;
        Ok(out)
    }

    /// Reuses `out` to return the `k` nearest points to `point_idx` excluding `point_idx`.
    pub fn k_nearest_excluding_self_into(
        &self,
        point_idx: usize,
        k: usize,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let q = self.point(point_idx)?;
        out.clear();
        if k == 0 || self.len() <= 1 {
            return Ok(());
        }
        self.knn_search(&q, k, Some(point_idx as u32), out);
        Ok(())
    }

    /// Returns the single nearest point `(index, dist_sq)` to `point_idx`, excluding `point_idx` itself.
    pub fn nearest_excluding_self(&self, point_idx: usize) -> CoreResult<Option<(usize, f64)>> {
        let q = self.point(point_idx)?;
        if self.len() <= 1 {
            return Ok(None);
        }
        let exclude = point_idx as u32;
        let mut best_dist_sq = f64::INFINITY;
        let mut best_idx = u32::MAX;

        let mut stack: Vec<(u32, f64)> = Vec::with_capacity(64);
        stack.push((0, 0.0));

        while let Some((node_idx, b_dist)) = stack.pop() {
            if b_dist > best_dist_sq {
                continue;
            }
            let node = &self.nodes[node_idx as usize];
            if node.left == u32::MAX {
                for entry in &self.entries[node.start..node.end] {
                    if entry.index == exclude {
                        continue;
                    }
                    let d2 = dist_sq(&q, &entry.coords);
                    if d2 < best_dist_sq || (d2 == best_dist_sq && entry.index < best_idx) {
                        best_dist_sq = d2;
                        best_idx = entry.index;
                    }
                }
                continue;
            }

            let left = node.left as usize;
            let right = node.right as usize;
            let dl = box_dist_sq(&q, &self.nodes[left]);
            let dr = box_dist_sq(&q, &self.nodes[right]);

            if q[node.split_dim] <= node.split_val {
                if dr <= best_dist_sq {
                    stack.push((right as u32, dr));
                }
                if dl <= best_dist_sq {
                    stack.push((left as u32, dl));
                }
            } else {
                if dl <= best_dist_sq {
                    stack.push((left as u32, dl));
                }
                if dr <= best_dist_sq {
                    stack.push((right as u32, dr));
                }
            }
        }

        if best_idx == u32::MAX {
            Ok(None)
        } else {
            Ok(Some((best_idx as usize, best_dist_sq)))
        }
    }

    /// Returns all points `(index, dist_sq)` within Euclidean `radius >= 0.0` of `query`,
    /// sorted by `(dist_sq, index)`.
    pub fn within_radius(&self, query: &[S; D], radius: f64) -> CoreResult<Vec<(usize, f64)>> {
        let mut out = Vec::new();
        self.within_radius_into(query, radius, &mut out)?;
        Ok(out)
    }

    /// Reuses `out` to return all points within Euclidean `radius >= 0.0` of `query`.
    pub fn within_radius_into(
        &self,
        query: &[S; D],
        radius: f64,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        if !radius.is_finite() {
            return Err(CoreError::non_finite("StaticKdTree::within_radius"));
        }
        if radius < 0.0 {
            return Err(CoreError::invalid_input("radius must be >= 0.0"));
        }
        self.within_radius_sq_into(query, radius * radius, out)
    }

    /// Returns all points `(index, dist_sq)` with `dist_sq <= radius_sq`, sorted by `(dist_sq, index)`.
    pub fn within_radius_sq(
        &self,
        query: &[S; D],
        radius_sq: f64,
    ) -> CoreResult<Vec<(usize, f64)>> {
        let mut out = Vec::new();
        self.within_radius_sq_into(query, radius_sq, &mut out)?;
        Ok(out)
    }

    /// Reuses `out` to return all points with `dist_sq <= radius_sq`.
    pub fn within_radius_sq_into(
        &self,
        query: &[S; D],
        radius_sq: f64,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let q = convert_query(query)?;
        if !radius_sq.is_finite() {
            return Err(CoreError::non_finite("StaticKdTree::within_radius_sq"));
        }
        if radius_sq < 0.0 {
            return Err(CoreError::invalid_input("radius_sq must be >= 0.0"));
        }
        out.clear();
        if self.nodes.is_empty() {
            return Ok(());
        }

        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(0);

        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx as usize];
            if box_dist_sq(&q, node) > radius_sq {
                continue;
            }
            if node.left == u32::MAX {
                for entry in &self.entries[node.start..node.end] {
                    let d2 = dist_sq(&q, &entry.coords);
                    if d2 <= radius_sq {
                        out.push((entry.index as usize, d2));
                    }
                }
            } else {
                stack.push(node.right);
                stack.push(node.left);
            }
        }

        out.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        Ok(())
    }

    fn build_node(&mut self, start: usize, end: usize) -> u32 {
        let mut min_bounds = self.entries[start].coords;
        let mut max_bounds = min_bounds;
        for entry in &self.entries[start + 1..end] {
            for d in 0..D {
                min_bounds[d] = min_bounds[d].min(entry.coords[d]);
                max_bounds[d] = max_bounds[d].max(entry.coords[d]);
            }
        }

        let node_idx = self.nodes.len() as u32;
        if end - start <= LEAF_SIZE {
            self.nodes.push(KdNode {
                min_bounds,
                max_bounds,
                start,
                end,
                split_dim: 0,
                split_val: 0.0,
                left: u32::MAX,
                right: u32::MAX,
            });
            return node_idx;
        }

        let mut split_dim = 0usize;
        let mut max_spread = -1.0_f64;
        for d in 0..D {
            let spread = max_bounds[d] - min_bounds[d];
            if spread > max_spread {
                max_spread = spread;
                split_dim = d;
            }
        }

        // Collapse identical/duplicate clusters into a single leaf to avoid empty splits.
        if max_spread <= 0.0 {
            self.nodes.push(KdNode {
                min_bounds,
                max_bounds,
                start,
                end,
                split_dim: 0,
                split_val: 0.0,
                left: u32::MAX,
                right: u32::MAX,
            });
            return node_idx;
        }

        let mid = start + (end - start) / 2;
        let slice = &mut self.entries[start..end];
        let local_mid = mid - start;
        slice.select_nth_unstable_by(local_mid, |a, b| {
            a.coords[split_dim]
                .total_cmp(&b.coords[split_dim])
                .then_with(|| a.index.cmp(&b.index))
        });
        let split_val = self.entries[mid].coords[split_dim];

        self.nodes.push(KdNode {
            min_bounds,
            max_bounds,
            start,
            end,
            split_dim,
            split_val,
            left: u32::MAX,
            right: u32::MAX,
        });

        let left = self.build_node(start, mid);
        let right = self.build_node(mid, end);
        self.nodes[node_idx as usize].left = left;
        self.nodes[node_idx as usize].right = right;
        node_idx
    }

    fn knn_search(
        &self,
        q: &[f64; D],
        k: usize,
        exclude: Option<u32>,
        out: &mut Vec<(usize, f64)>,
    ) {
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k);
        let mut stack: Vec<(u32, f64)> = Vec::with_capacity(64);
        stack.push((0, 0.0));

        while let Some((node_idx, b_dist)) = stack.pop() {
            if heap.len() == k {
                // Use strict `>` so tied distances in adjacent boxes are always explored.
                if let Some(&worst) = heap.peek()
                    && b_dist > worst.dist_sq
                {
                    continue;
                }
            }

            let node = &self.nodes[node_idx as usize];
            if node.left == u32::MAX {
                for entry in &self.entries[node.start..node.end] {
                    if Some(entry.index) == exclude {
                        continue;
                    }
                    let d2 = dist_sq(q, &entry.coords);
                    let cand = Candidate {
                        dist_sq: d2,
                        index: entry.index,
                    };
                    if heap.len() < k {
                        heap.push(cand);
                    } else if let Some(&worst) = heap.peek()
                        && cand < worst
                    {
                        heap.pop();
                        heap.push(cand);
                    }
                }
                continue;
            }

            let left = node.left as usize;
            let right = node.right as usize;
            let dl = box_dist_sq(q, &self.nodes[left]);
            let dr = box_dist_sq(q, &self.nodes[right]);
            let thresh = if heap.len() == k {
                heap.peek().map_or(f64::INFINITY, |w| w.dist_sq)
            } else {
                f64::INFINITY
            };

            if q[node.split_dim] <= node.split_val {
                if dr <= thresh {
                    stack.push((right as u32, dr));
                }
                if dl <= thresh {
                    stack.push((left as u32, dl));
                }
            } else {
                if dl <= thresh {
                    stack.push((left as u32, dl));
                }
                if dr <= thresh {
                    stack.push((right as u32, dr));
                }
            }
        }

        let sorted = heap.into_sorted_vec();
        out.extend(sorted.into_iter().map(|c| (c.index as usize, c.dist_sq)));
    }
}

impl<S: Scalar> StaticKdTree<2, S> {
    /// Builds a 2D KD-tree directly from an SoA [`Layout2d<S>`].
    pub fn from_layout(layout: &Layout2d<S>) -> CoreResult<Self> {
        let n = layout.len();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_input(
                "StaticKdTree point count exceeds u32::MAX",
            ));
        }
        layout.validate_finite()?;

        let xs = layout.x();
        let ys = layout.y();
        let mut coords_by_id = Vec::with_capacity(n);
        let mut entries = Vec::with_capacity(n);
        for i in 0..n {
            let c = [xs[i].to_f64(), ys[i].to_f64()];
            coords_by_id.push(c);
            entries.push(KdEntry {
                coords: c,
                index: i as u32,
            });
        }

        let mut tree = Self {
            nodes: Vec::new(),
            entries,
            coords_by_id,
            _marker: PhantomData,
        };
        if n > 0 {
            tree.nodes.reserve((2 * n / LEAF_SIZE).max(8));
            tree.build_node(0, n);
        }
        Ok(tree)
    }
}

#[inline]
fn convert_query<const D: usize, S: Scalar>(query: &[S; D]) -> CoreResult<[f64; D]> {
    let mut out = [0.0_f64; D];
    for d in 0..D {
        if !query[d].is_finite() {
            return Err(CoreError::non_finite("StaticKdTree query"));
        }
        out[d] = query[d].to_f64();
    }
    Ok(out)
}

#[inline]
fn dist_sq<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    let mut sum = 0.0_f64;
    for d in 0..D {
        let diff = a[d] - b[d];
        sum += diff * diff;
    }
    sum
}

#[inline]
fn box_dist_sq<const D: usize>(q: &[f64; D], node: &KdNode<D>) -> f64 {
    let mut sum = 0.0_f64;
    for d in 0..D {
        let v = q[d];
        let diff = if v < node.min_bounds[d] {
            node.min_bounds[d] - v
        } else if v > node.max_bounds[d] {
            v - node.max_bounds[d]
        } else {
            0.0
        };
        sum += diff * diff;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use egraph_core::DeterministicRng;

    fn brute_knn<const D: usize>(
        pts: &[[f64; D]],
        q: &[f64; D],
        k: usize,
        exclude: Option<usize>,
    ) -> Vec<(usize, f64)> {
        let mut all: Vec<(usize, f64)> = pts
            .iter()
            .enumerate()
            .filter(|(i, _)| Some(*i) != exclude)
            .map(|(i, p)| (i, dist_sq(q, p)))
            .collect();
        all.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        all.truncate(k);
        all
    }

    fn brute_within<const D: usize>(pts: &[[f64; D]], q: &[f64; D], r2: f64) -> Vec<(usize, f64)> {
        let mut all: Vec<(usize, f64)> = pts
            .iter()
            .enumerate()
            .map(|(i, p)| (i, dist_sq(q, p)))
            .filter(|&(_, d2)| d2 <= r2)
            .collect();
        all.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        all
    }

    #[test]
    fn test_kdtree_exact_parity_random_collinear_and_duplicates() {
        // 1. Random 2D points (N > LEAF_SIZE so multiple tree levels are exercised).
        let mut rng = DeterministicRng::from_seed_u64(1337);
        let random_pts: Vec<[f64; 2]> = (0..120)
            .map(|_| [rng.next_f64() * 20.0 - 10.0, rng.next_f64() * 20.0 - 10.0])
            .collect();

        // 2. Collinear integer grid with abundant distance ties.
        let mut collinear_pts = Vec::new();
        for i in 0..45 {
            collinear_pts.push([(i % 15) as f64, 3.0]);
        }

        // 3. Heavy duplicates + isolated outliers.
        let mut duplicate_pts = vec![[1.5, -2.5]; 40];
        duplicate_pts.extend(vec![[-3.0, 4.0]; 20]);
        duplicate_pts.push([0.0, 0.0]);

        for pts in [&random_pts, &collinear_pts, &duplicate_pts] {
            let tree = StaticKdTree::<2, f64>::build(pts).unwrap();
            assert_eq!(tree.len(), pts.len());

            for i in 0..pts.len() {
                for &k in &[1usize, 5, 20, 60] {
                    let got = tree.k_nearest_excluding_self(i, k).unwrap();
                    let expected = brute_knn(pts, &pts[i], k, Some(i));
                    assert_eq!(got, expected, "k={k}, i={i}");
                }
                let nearest = tree.nearest_excluding_self(i).unwrap();
                let expected_1 = brute_knn(pts, &pts[i], 1, Some(i)).into_iter().next();
                assert_eq!(nearest, expected_1);
            }

            let queries = [[0.0, 0.0], [1.5, -2.5], [7.0, 3.0], [-12.0, 12.0]];
            for q in &queries {
                assert_eq!(tree.k_nearest(q, 10).unwrap(), brute_knn(pts, q, 10, None));
                assert_eq!(
                    tree.within_radius(q, 3.0).unwrap(),
                    brute_within(pts, q, 9.0)
                );
                assert_eq!(
                    tree.within_radius_sq(q, 0.0).unwrap(),
                    brute_within(pts, q, 0.0)
                );
            }
        }
    }

    #[test]
    fn test_kdtree_3d_layout_and_error_cases() {
        let layout3 = LayoutNd::from_flat(
            4,
            3,
            vec![
                0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0,
            ],
        )
        .unwrap();
        let tree3 = StaticKdTree::<3, f32>::from_layout_nd(&layout3).unwrap();
        let nn = tree3.nearest_excluding_self(0).unwrap().unwrap();
        assert_eq!(nn, (1, 1.0));

        let empty = StaticKdTree::<2, f64>::build(&[]).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.k_nearest(&[0.0, 0.0], 3).unwrap(), Vec::new());

        let single = StaticKdTree::<2, f64>::build(&[[2.0, 3.0]]).unwrap();
        assert_eq!(single.nearest_excluding_self(0).unwrap(), None);

        assert!(matches!(
            StaticKdTree::<0, f64>::build(&[]),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            StaticKdTree::<2, f64>::build(&[[f64::NAN, 0.0]]),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            single.within_radius(&[0.0, 0.0], -1.0),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            single.nearest_excluding_self(1),
            Err(CoreError::IndexOutOfBounds { .. })
        ));
    }
}
