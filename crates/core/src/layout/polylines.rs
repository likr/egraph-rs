use std::ops::Range;

use crate::csr::CsrGraph;
use crate::error::{CoreError, CoreResult};
use crate::geometry::{EuclideanLayout, EuclideanStorageMut, EuclideanStorageRef, LayoutStorage};
use crate::scalar::Scalar;

/// Contiguous polyline buffer representing edge routes across arbitrary [`LayoutStorage`] spaces.
///
/// Stores `edge_offsets` (length `num_edges + 1`) alongside a contiguous control-point buffer
/// `points: L` such that edge `e` owns points `edge_offsets[e]..edge_offsets[e + 1]`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EdgePolylines<L: LayoutStorage> {
    edge_offsets: Vec<u32>,
    points: L,
}

fn validate_offsets<L: LayoutStorage>(edge_offsets: &[u32], points: &L) -> CoreResult<()> {
    if edge_offsets.is_empty() {
        return Err(CoreError::invalid_input(
            "EdgePolylines edge_offsets must contain at least 1 element",
        ));
    }
    if edge_offsets[0] != 0 {
        return Err(CoreError::invalid_input(format!(
            "EdgePolylines edge_offsets[0] must be 0, got {}",
            edge_offsets[0]
        )));
    }
    for (edge, win) in edge_offsets.windows(2).enumerate() {
        if win[0] > win[1] {
            return Err(CoreError::invalid_input(format!(
                "EdgePolylines edge_offsets must be monotonic at edge {edge}: {} > {}",
                win[0], win[1]
            )));
        }
    }
    let last = *edge_offsets.last().unwrap() as usize;
    if last != points.len() {
        return Err(CoreError::dimension_mismatch(
            last,
            points.len(),
            "EdgePolylines::edge_offsets",
        ));
    }
    Ok(())
}

impl<L: LayoutStorage> EdgePolylines<L> {
    /// Constructs [`EdgePolylines`] by copying `edge_offsets` and taking ownership of `points`.
    pub fn from_parts(edge_offsets: &[u32], points: L) -> CoreResult<Self> {
        Self::from_parts_vec(edge_offsets.to_vec(), points)
    }

    /// Constructs [`EdgePolylines`] from owned `edge_offsets` and `points`, validating offset
    /// invariants (`edge_offsets[0] == 0`, monotonic non-decreasing, `last == points.len()`).
    pub fn from_parts_vec(edge_offsets: Vec<u32>, points: L) -> CoreResult<Self> {
        validate_offsets(&edge_offsets, &points)?;
        Ok(Self {
            edge_offsets,
            points,
        })
    }

    /// Consumes `self` and returns `(edge_offsets, points)`.
    #[inline]
    pub fn into_parts(self) -> (Vec<u32>, L) {
        (self.edge_offsets, self.points)
    }

    /// Replaces both `edge_offsets` and `points` atomically after validating invariants.
    pub fn replace_parts(&mut self, edge_offsets: Vec<u32>, points: L) -> CoreResult<()> {
        validate_offsets(&edge_offsets, &points)?;
        self.edge_offsets = edge_offsets;
        self.points = points;
        Ok(())
    }

    /// Returns the number of polylines (logical edges) stored in the buffer.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.edge_offsets.len().saturating_sub(1)
    }

    /// Returns the total number of control points across all polylines.
    #[inline]
    pub fn total_points(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` if the buffer stores zero polylines.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_edges() == 0
    }

    /// Returns the edge offset slice of length `num_edges() + 1`.
    #[inline]
    pub fn edge_offsets(&self) -> &[u32] {
        &self.edge_offsets
    }

    /// Returns the range of control-point indices belonging to `edge`.
    #[inline]
    pub fn edge_point_range(&self, edge: usize) -> CoreResult<Range<usize>> {
        let num_edges = self.num_edges();
        if edge >= num_edges {
            return Err(CoreError::index_out_of_bounds(
                edge,
                num_edges,
                "EdgePolylines::edge_point_range",
            ));
        }
        let start = self.edge_offsets[edge] as usize;
        let end = self.edge_offsets[edge + 1] as usize;
        Ok(start..end)
    }

    /// Returns an immutable reference to the contiguous control-point buffer.
    #[inline]
    pub fn points(&self) -> &L {
        &self.points
    }

    /// Returns a mutable reference to the contiguous control-point buffer.
    #[inline]
    pub fn points_mut(&mut self) -> &mut L {
        &mut self.points
    }

    /// Validates that every control-point coordinate in the buffer is finite.
    #[inline]
    pub fn validate_finite(&self) -> CoreResult<()> {
        self.points.validate_finite()
    }
}

impl<L: EuclideanLayout> EdgePolylines<L> {
    /// Initializes 2-point straight-line edge polylines from `graph` and node positions `layout`.
    ///
    /// Uses `graph.edges()` (all `(u, v)` rows for directed graphs; canonical `u < v` pairs
    /// for undirected graphs).
    pub fn straight_line_from_graph<W>(graph: &CsrGraph<W>, layout: &L) -> CoreResult<Self> {
        if layout.len() != graph.num_nodes() {
            return Err(CoreError::dimension_mismatch(
                graph.num_nodes(),
                layout.len(),
                "EdgePolylines::straight_line_from_graph",
            ));
        }
        layout.validate_finite()?;

        let edges = graph.edges();
        let num_edges = edges.len();
        let total_points = num_edges
            .checked_mul(2)
            .ok_or_else(|| CoreError::invalid_input("EdgePolylines point count overflow"))?;
        if u32::try_from(total_points).is_err() {
            return Err(CoreError::invalid_input(
                "EdgePolylines total_points exceeds u32::MAX",
            ));
        }

        let mut edge_offsets = Vec::with_capacity(num_edges + 1);
        edge_offsets.push(0u32);
        for e in 0..num_edges {
            edge_offsets.push(((e + 1) * 2) as u32);
        }

        let mut points = layout.zeros_like(total_points)?;
        copy_straight_endpoints(
            layout.as_euclidean_storage(),
            points.as_euclidean_storage_mut(),
            &edges,
        )?;
        Ok(Self {
            edge_offsets,
            points,
        })
    }

    /// Returns a new [`EdgePolylines`] with `interior_points_per_segment` linearly interpolated
    /// points inserted along every segment of every polyline containing at least 2 points.
    pub fn with_subdivisions(&self, interior_points_per_segment: usize) -> CoreResult<Self> {
        self.points.validate_finite()?;

        let num_edges = self.num_edges();
        let mut new_offsets = Vec::with_capacity(num_edges + 1);
        new_offsets.push(0u32);
        let mut total_new_points = 0usize;

        for e in 0..num_edges {
            let start = self.edge_offsets[e] as usize;
            let end = self.edge_offsets[e + 1] as usize;
            let m = end - start;
            let new_m = if m <= 1 {
                m
            } else {
                let extra = (m - 1)
                    .checked_mul(interior_points_per_segment)
                    .ok_or_else(|| CoreError::invalid_input("subdivision point count overflow"))?;
                m.checked_add(extra)
                    .ok_or_else(|| CoreError::invalid_input("subdivision point count overflow"))?
            };
            total_new_points = total_new_points
                .checked_add(new_m)
                .ok_or_else(|| CoreError::invalid_input("subdivision point count overflow"))?;
            let off = u32::try_from(total_new_points).map_err(|_| {
                CoreError::invalid_input("EdgePolylines total_points exceeds u32::MAX")
            })?;
            new_offsets.push(off);
        }

        let mut new_points = self.points.zeros_like(total_new_points)?;
        subdivide_storage(
            self.points.as_euclidean_storage(),
            new_points.as_euclidean_storage_mut(),
            &self.edge_offsets,
            interior_points_per_segment,
        )?;
        Ok(Self {
            edge_offsets: new_offsets,
            points: new_points,
        })
    }
}

fn copy_straight_endpoints<S: Scalar>(
    src: EuclideanStorageRef<'_, S>,
    dst: EuclideanStorageMut<'_, S>,
    edges: &[(u32, u32)],
) -> CoreResult<()> {
    match (src, dst) {
        (
            EuclideanStorageRef::Soa2d { x: sx, y: sy },
            EuclideanStorageMut::Soa2d { x: dx, y: dy },
        ) => {
            for (e, &(u, v)) in edges.iter().enumerate() {
                let ui = u as usize;
                let vi = v as usize;
                dx[2 * e] = sx[ui];
                dy[2 * e] = sy[ui];
                dx[2 * e + 1] = sx[vi];
                dy[2 * e + 1] = sy[vi];
            }
            Ok(())
        }
        (
            EuclideanStorageRef::Soa3d {
                x: sx,
                y: sy,
                z: sz,
            },
            EuclideanStorageMut::Soa3d {
                x: dx,
                y: dy,
                z: dz,
            },
        ) => {
            for (e, &(u, v)) in edges.iter().enumerate() {
                let ui = u as usize;
                let vi = v as usize;
                dx[2 * e] = sx[ui];
                dy[2 * e] = sy[ui];
                dz[2 * e] = sz[ui];
                dx[2 * e + 1] = sx[vi];
                dy[2 * e + 1] = sy[vi];
                dz[2 * e + 1] = sz[vi];
            }
            Ok(())
        }
        (
            EuclideanStorageRef::FlatNd {
                coords: sc,
                dims: sd,
                ..
            },
            EuclideanStorageMut::FlatNd {
                coords: dc,
                dims: dd,
                ..
            },
        ) if sd == dd => {
            for (e, &(u, v)) in edges.iter().enumerate() {
                let us = (u as usize) * sd;
                let vs = (v as usize) * sd;
                let p0 = (2 * e) * sd;
                let p1 = (2 * e + 1) * sd;
                dc[p0..p0 + sd].copy_from_slice(&sc[us..us + sd]);
                dc[p1..p1 + sd].copy_from_slice(&sc[vs..vs + sd]);
            }
            Ok(())
        }
        _ => Err(CoreError::invalid_input(
            "mismatched Euclidean storage representation in EdgePolylines",
        )),
    }
}

fn subdivide_storage<S: Scalar>(
    src: EuclideanStorageRef<'_, S>,
    dst: EuclideanStorageMut<'_, S>,
    edge_offsets: &[u32],
    interior_points_per_segment: usize,
) -> CoreResult<()> {
    let denom = (interior_points_per_segment + 1) as f64;
    let num_edges = edge_offsets.len() - 1;
    match (src, dst) {
        (
            EuclideanStorageRef::Soa2d { x: sx, y: sy },
            EuclideanStorageMut::Soa2d { x: dx, y: dy },
        ) => {
            let mut out = 0usize;
            for e in 0..num_edges {
                let start = edge_offsets[e] as usize;
                let end = edge_offsets[e + 1] as usize;
                let m = end - start;
                if m == 0 {
                    continue;
                }
                for seg in 0..(m - 1) {
                    let i0 = start + seg;
                    let i1 = i0 + 1;
                    let x0 = sx[i0].to_f64();
                    let y0 = sy[i0].to_f64();
                    let x1 = sx[i1].to_f64();
                    let y1 = sy[i1].to_f64();
                    dx[out] = sx[i0];
                    dy[out] = sy[i0];
                    out += 1;
                    for k in 1..=interior_points_per_segment {
                        let t = (k as f64) / denom;
                        dx[out] = S::from_f64(x0 + t * (x1 - x0));
                        dy[out] = S::from_f64(y0 + t * (y1 - y0));
                        out += 1;
                    }
                }
                dx[out] = sx[end - 1];
                dy[out] = sy[end - 1];
                out += 1;
            }
            Ok(())
        }
        (
            EuclideanStorageRef::Soa3d {
                x: sx,
                y: sy,
                z: sz,
            },
            EuclideanStorageMut::Soa3d {
                x: dx,
                y: dy,
                z: dz,
            },
        ) => {
            let mut out = 0usize;
            for e in 0..num_edges {
                let start = edge_offsets[e] as usize;
                let end = edge_offsets[e + 1] as usize;
                let m = end - start;
                if m == 0 {
                    continue;
                }
                for seg in 0..(m - 1) {
                    let i0 = start + seg;
                    let i1 = i0 + 1;
                    let x0 = sx[i0].to_f64();
                    let y0 = sy[i0].to_f64();
                    let z0 = sz[i0].to_f64();
                    let x1 = sx[i1].to_f64();
                    let y1 = sy[i1].to_f64();
                    let z1 = sz[i1].to_f64();
                    dx[out] = sx[i0];
                    dy[out] = sy[i0];
                    dz[out] = sz[i0];
                    out += 1;
                    for k in 1..=interior_points_per_segment {
                        let t = (k as f64) / denom;
                        dx[out] = S::from_f64(x0 + t * (x1 - x0));
                        dy[out] = S::from_f64(y0 + t * (y1 - y0));
                        dz[out] = S::from_f64(z0 + t * (z1 - z0));
                        out += 1;
                    }
                }
                dx[out] = sx[end - 1];
                dy[out] = sy[end - 1];
                dz[out] = sz[end - 1];
                out += 1;
            }
            Ok(())
        }
        (
            EuclideanStorageRef::FlatNd {
                coords: sc,
                dims: sd,
                ..
            },
            EuclideanStorageMut::FlatNd {
                coords: dc,
                dims: dd,
                ..
            },
        ) if sd == dd => {
            let mut out = 0usize;
            for e in 0..num_edges {
                let start = edge_offsets[e] as usize;
                let end = edge_offsets[e + 1] as usize;
                let m = end - start;
                if m == 0 {
                    continue;
                }
                for seg in 0..(m - 1) {
                    let s0 = (start + seg) * sd;
                    let s1 = s0 + sd;
                    let dst0 = out * sd;
                    dc[dst0..dst0 + sd].copy_from_slice(&sc[s0..s0 + sd]);
                    out += 1;
                    for k in 1..=interior_points_per_segment {
                        let t = (k as f64) / denom;
                        let dstk = out * sd;
                        for axis in 0..sd {
                            let a = sc[s0 + axis].to_f64();
                            let b = sc[s1 + axis].to_f64();
                            dc[dstk + axis] = S::from_f64(a + t * (b - a));
                        }
                        out += 1;
                    }
                }
                let s_last = (end - 1) * sd;
                let dst_last = out * sd;
                dc[dst_last..dst_last + sd].copy_from_slice(&sc[s_last..s_last + sd]);
                out += 1;
            }
            Ok(())
        }
        _ => Err(CoreError::invalid_input(
            "mismatched Euclidean storage representation in EdgePolylines",
        )),
    }
}
