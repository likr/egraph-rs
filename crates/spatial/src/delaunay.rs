//! Optional 2D Delaunay triangulation wrapper built on `spade` (`feature = "delaunay"`).

use std::cmp::Ordering;

use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, Scalar};
use spade::{DelaunayTriangulation, HasPosition, Point2, Triangulation};

#[derive(Debug, Clone, Copy)]
struct IndexedPoint2 {
    index: u32,
    x: f64,
    y: f64,
}

impl HasPosition for IndexedPoint2 {
    type Scalar = f64;

    #[inline]
    fn position(&self) -> Point2<f64> {
        Point2::new(self.x, self.y)
    }
}

/// Deterministic 2D Delaunay triangulation extracted from a [`Layout2d<S>`].
///
/// Deduplicates coincident coordinates deterministically (retaining the smallest vertex ID
/// per unique site), tolerates collinear point sets without panicking, and returns sorted
/// `u32` undirected edges and canonical triangles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Delaunay2d {
    num_nodes: usize,
    edges: Vec<(u32, u32)>,
    triangles: Vec<[u32; 3]>,
}

impl Delaunay2d {
    /// Computes the 2D Delaunay triangulation of `layout`.
    ///
    /// Rejects non-finite coordinates with [`CoreError::NonFiniteValue`] and maps `spade`
    /// geometric range violations to [`CoreError::NumericalBreakdown`].
    pub fn triangulate<S: Scalar>(layout: &Layout2d<S>) -> CoreResult<Self> {
        layout.validate_finite()?;
        let n = layout.len();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_input(
                "Delaunay2d vertex count exceeds u32::MAX",
            ));
        }
        if n < 2 {
            return Ok(Self {
                num_nodes: n,
                edges: Vec::new(),
                triangles: Vec::new(),
            });
        }

        let xs = layout.x();
        let ys = layout.y();
        let mut pts: Vec<IndexedPoint2> = (0..n)
            .map(|i| IndexedPoint2 {
                index: i as u32,
                x: xs[i].to_f64(),
                y: ys[i].to_f64(),
            })
            .collect();

        // Deduplicate coincident sites deterministically, keeping the smallest vertex ID.
        pts.sort_unstable_by(|a, b| {
            a.x.total_cmp(&b.x)
                .then_with(|| a.y.total_cmp(&b.y))
                .then_with(|| a.index.cmp(&b.index))
        });
        pts.dedup_by(|b, a| a.x == b.x && a.y == b.y);

        // Re-sort unique sites by original vertex index so insertion order is index-deterministic.
        pts.sort_unstable_by_key(|p| p.index);

        if pts.len() < 2 {
            return Ok(Self {
                num_nodes: n,
                edges: Vec::new(),
                triangles: Vec::new(),
            });
        }

        let mut dt = DelaunayTriangulation::<IndexedPoint2>::with_capacity(pts.len(), 0, 0);
        for p in pts {
            dt.insert(p).map_err(|err| {
                CoreError::numerical_breakdown(format!("spade Delaunay insertion failed: {err}"))
            })?;
        }

        let mut edges: Vec<(u32, u32)> = dt
            .undirected_edges()
            .map(|e| {
                let [v0, v1] = e.vertices();
                let a = v0.data().index;
                let b = v1.data().index;
                if a < b { (a, b) } else { (b, a) }
            })
            .collect();
        edges.sort_unstable();
        edges.dedup();

        let mut triangles: Vec<[u32; 3]> = dt
            .inner_faces()
            .map(|f| {
                let [v0, v1, v2] = f.vertices();
                canonical_triangle([v0.data().index, v1.data().index, v2.data().index])
            })
            .collect();
        triangles.sort_unstable();

        Ok(Self {
            num_nodes: n,
            edges,
            triangles,
        })
    }

    /// Extracts sorted canonical undirected Delaunay edges `(u, v)` (`u < v`) into `out_edges`
    /// without building or sorting faces.
    ///
    /// Uses `spade::DelaunayTriangulation::bulk_load_stable` directly on `(x, y)`-sorted
    /// deduplicated sites for spatial locality and $O(N \log N)$ throughput.
    pub fn triangulate_edges_into<S: Scalar>(
        layout: &Layout2d<S>,
        out_edges: &mut Vec<(u32, u32)>,
    ) -> CoreResult<()> {
        out_edges.clear();
        layout.validate_finite()?;
        let n = layout.len();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_input(
                "Delaunay2d vertex count exceeds u32::MAX",
            ));
        }
        if n < 2 {
            return Ok(());
        }

        let xs = layout.x();
        let ys = layout.y();
        let mut pts: Vec<IndexedPoint2> = (0..n)
            .map(|i| IndexedPoint2 {
                index: i as u32,
                x: xs[i].to_f64(),
                y: ys[i].to_f64(),
            })
            .collect();

        // Deduplicate coincident sites deterministically, keeping the smallest vertex ID.
        pts.sort_unstable_by(|a, b| {
            a.x.total_cmp(&b.x)
                .then_with(|| a.y.total_cmp(&b.y))
                .then_with(|| a.index.cmp(&b.index))
        });
        pts.dedup_by(|b, a| a.x == b.x && a.y == b.y);

        if pts.len() < 2 {
            return Ok(());
        }

        let dt = DelaunayTriangulation::<IndexedPoint2>::bulk_load_stable(pts).map_err(|err| {
            CoreError::numerical_breakdown(format!("spade Delaunay bulk_load_stable failed: {err}"))
        })?;

        out_edges.reserve(dt.num_undirected_edges());
        for e in dt.undirected_edges() {
            let [v0, v1] = e.vertices();
            let a = v0.data().index;
            let b = v1.data().index;
            out_edges.push(if a < b { (a, b) } else { (b, a) });
        }
        out_edges.sort_unstable();
        out_edges.dedup();
        Ok(())
    }

    /// Returns the number of vertices in the source layout.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns sorted canonical undirected triangulation edges `(u, v)` with `u < v`.
    #[inline]
    pub fn edges(&self) -> &[(u32, u32)] {
        &self.edges
    }

    /// Returns sorted canonical Delaunay triangles `[a, b, c]` (rotated so `a` is smallest,
    /// preserving orientation).
    #[inline]
    pub fn triangles(&self) -> &[[u32; 3]] {
        &self.triangles
    }

    /// Consumes `self` and returns the edge list.
    #[inline]
    pub fn into_edges(self) -> Vec<(u32, u32)> {
        self.edges
    }

    /// Converts the triangulation edges into an unweighted undirected [`CsrGraph<f64>`].
    pub fn to_csr(&self) -> CoreResult<CsrGraph<f64>> {
        CsrGraph::from_edges(self.num_nodes, &self.edges, None, false, false)
    }
}

#[inline]
fn canonical_triangle(tri: [u32; 3]) -> [u32; 3] {
    let min_pos = match tri[0].cmp(&tri[1]) {
        Ordering::Less | Ordering::Equal => {
            if tri[0] <= tri[2] {
                0
            } else {
                2
            }
        }
        Ordering::Greater => {
            if tri[1] <= tri[2] {
                1
            } else {
                2
            }
        }
    };
    match min_pos {
        0 => tri,
        1 => [tri[1], tri[2], tri[0]],
        _ => [tri[2], tri[0], tri[1]],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delaunay_square_triangle_collinear_and_coincident() {
        let square =
            Layout2d::from_xy(vec![0.0_f64, 1.0, 0.0, 1.0], vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let dt_sq = Delaunay2d::triangulate(&square).unwrap();
        assert_eq!(dt_sq.num_nodes(), 4);
        assert_eq!(dt_sq.edges().len(), 5);
        assert_eq!(dt_sq.triangles().len(), 2);
        let csr = dt_sq.to_csr().unwrap();
        assert_eq!(csr.num_edges(), 5);

        let collinear =
            Layout2d::from_xy(vec![0.0_f32, 1.0, 2.0, 3.0], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let dt_col = Delaunay2d::triangulate(&collinear).unwrap();
        assert_eq!(dt_col.edges(), &[(0, 1), (1, 2), (2, 3)]);
        assert!(dt_col.triangles().is_empty());

        // Coincident points keep smallest vertex IDs deterministically and do not panic.
        let dup =
            Layout2d::from_xy(vec![0.0_f64, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]).unwrap();
        let dt_dup = Delaunay2d::triangulate(&dup).unwrap();
        assert_eq!(dt_dup.edges(), &[(0, 2), (0, 3), (2, 3)]);
        assert_eq!(dt_dup.triangles(), &[[0, 2, 3]]);

        let single = Layout2d::from_xy(vec![0.0_f64], vec![0.0]).unwrap();
        let dt_single = Delaunay2d::triangulate(&single).unwrap();
        assert!(dt_single.edges().is_empty());
        assert!(dt_single.triangles().is_empty());
    }

    #[test]
    fn test_delaunay_rejects_nonfinite_and_extreme_coords() {
        let bad = Layout2d::from_xy(vec![0.0, f64::NAN, 1.0], vec![0.0, 1.0, 0.0]).unwrap();
        assert!(matches!(
            Delaunay2d::triangulate(&bad),
            Err(CoreError::NonFiniteValue { .. })
        ));

        let huge = Layout2d::from_xy(vec![0.0, 1e200, 1.0], vec![0.0, 0.0, 1.0]).unwrap();
        assert!(matches!(
            Delaunay2d::triangulate(&huge),
            Err(CoreError::NumericalBreakdown(_))
        ));
    }

    #[test]
    fn test_delaunay_triangulate_edges_into_matches_and_errors() {
        let mut edges = vec![(99, 100)];

        let square =
            Layout2d::from_xy(vec![0.0_f64, 1.0, 0.0, 1.0], vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        Delaunay2d::triangulate_edges_into(&square, &mut edges).unwrap();
        assert_eq!(edges.len(), 5);
        assert!(edges.windows(2).all(|w| w[0] < w[1]));

        let collinear =
            Layout2d::from_xy(vec![0.0_f32, 1.0, 2.0, 3.0], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        Delaunay2d::triangulate_edges_into(&collinear, &mut edges).unwrap();
        assert_eq!(edges, vec![(0, 1), (1, 2), (2, 3)]);

        let dup =
            Layout2d::from_xy(vec![0.0_f64, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]).unwrap();
        Delaunay2d::triangulate_edges_into(&dup, &mut edges).unwrap();
        assert_eq!(edges, vec![(0, 2), (0, 3), (2, 3)]);

        let single = Layout2d::from_xy(vec![0.0_f64], vec![0.0]).unwrap();
        Delaunay2d::triangulate_edges_into(&single, &mut edges).unwrap();
        assert!(edges.is_empty());

        let bad = Layout2d::from_xy(vec![0.0, f64::NAN, 1.0], vec![0.0, 1.0, 0.0]).unwrap();
        assert!(matches!(
            Delaunay2d::triangulate_edges_into(&bad, &mut edges),
            Err(CoreError::NonFiniteValue { .. })
        ));

        let huge = Layout2d::from_xy(vec![0.0, 1e200, 1.0], vec![0.0, 0.0, 1.0]).unwrap();
        assert!(matches!(
            Delaunay2d::triangulate_edges_into(&huge, &mut edges),
            Err(CoreError::NumericalBreakdown(_))
        ));
    }
}
