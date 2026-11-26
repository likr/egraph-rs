use egraph_core::{CoreError, CoreResult, LayoutNd, Scalar};

use super::{ArenaNode3d, ArenaOctree3d, COINCIDENT_DIST_SQ};

enum CoordSource<'a, S: Scalar> {
    Xyz {
        xs: &'a [S],
        ys: &'a [S],
        zs: &'a [S],
    },
    Flat {
        coords: &'a [S],
        num_nodes: usize,
    },
}

impl<S: Scalar> ArenaOctree3d<S> {
    /// Rebuilds the 3D octree from a 3D [`LayoutNd<S>`] with unit masses (`1.0`).
    ///
    /// Returns [`CoreError::DimensionMismatch`] when `layout.dims() != 3`.
    pub fn build(&mut self, layout: &LayoutNd<S>) -> CoreResult<()> {
        if layout.dims() != 3 {
            return Err(CoreError::dimension_mismatch(
                3,
                layout.dims(),
                "ArenaOctree3d::build",
            ));
        }
        self.build_flat(layout.coords(), layout.num_nodes())
    }

    /// Rebuilds the 3D octree from separate `x`, `y`, and `z` slices with unit masses (`1.0`).
    pub fn build_xyz(&mut self, x: &[S], y: &[S], z: &[S]) -> CoreResult<()> {
        self.build_inner(
            CoordSource::Xyz {
                xs: x,
                ys: y,
                zs: z,
            },
            None,
            "ArenaOctree3d::build_xyz",
        )
    }

    /// Rebuilds the weighted 3D octree from a 3D [`LayoutNd<S>`] and per-point `masses`.
    ///
    /// Returns [`CoreError::DimensionMismatch`] when `layout.dims() != 3` or `masses.len() != layout.num_nodes()`.
    pub fn build_weighted(&mut self, layout: &LayoutNd<S>, masses: &[f64]) -> CoreResult<()> {
        if layout.dims() != 3 {
            return Err(CoreError::dimension_mismatch(
                3,
                layout.dims(),
                "ArenaOctree3d::build_weighted",
            ));
        }
        self.build_weighted_flat(layout.coords(), layout.num_nodes(), masses)
    }

    /// Rebuilds the weighted 3D octree from separate `x`, `y`, `z`, and `masses` slices.
    pub fn build_weighted_xyz(
        &mut self,
        x: &[S],
        y: &[S],
        z: &[S],
        masses: &[f64],
    ) -> CoreResult<()> {
        self.build_inner(
            CoordSource::Xyz {
                xs: x,
                ys: y,
                zs: z,
            },
            Some(masses),
            "ArenaOctree3d::build_weighted_xyz",
        )
    }

    /// Rebuilds the 3D octree from row-major flat coordinates `[x0, y0, z0, ...]` with unit masses (`1.0`).
    pub fn build_flat(&mut self, coords: &[S], num_nodes: usize) -> CoreResult<()> {
        self.build_inner(
            CoordSource::Flat { coords, num_nodes },
            None,
            "ArenaOctree3d::build_flat",
        )
    }

    /// Rebuilds the weighted 3D octree from row-major flat coordinates `[x0, y0, z0, ...]` and `masses`.
    pub fn build_weighted_flat(
        &mut self,
        coords: &[S],
        num_nodes: usize,
        masses: &[f64],
    ) -> CoreResult<()> {
        self.build_inner(
            CoordSource::Flat { coords, num_nodes },
            Some(masses),
            "ArenaOctree3d::build_weighted_flat",
        )
    }

    fn build_inner(
        &mut self,
        source: CoordSource<'_, S>,
        masses: Option<&[f64]>,
        context: &'static str,
    ) -> CoreResult<()> {
        let n = match source {
            CoordSource::Xyz { xs, ys, zs } => {
                if xs.len() != ys.len() {
                    return Err(CoreError::dimension_mismatch(xs.len(), ys.len(), context));
                }
                if xs.len() != zs.len() {
                    return Err(CoreError::dimension_mismatch(xs.len(), zs.len(), context));
                }
                xs.len()
            }
            CoordSource::Flat { coords, num_nodes } => {
                let expected = num_nodes
                    .checked_mul(3)
                    .ok_or_else(|| CoreError::invalid_input("num_nodes * 3 overflows usize"))?;
                if coords.len() != expected {
                    return Err(CoreError::dimension_mismatch(
                        expected,
                        coords.len(),
                        context,
                    ));
                }
                num_nodes
            }
        };

        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_input(
                "ArenaOctree3d point count exceeds u32::MAX",
            ));
        }

        if let Some(m) = masses {
            if m.len() != n {
                return Err(CoreError::dimension_mismatch(n, m.len(), context));
            }
            for &w in m {
                if !w.is_finite() {
                    return Err(CoreError::non_finite(context));
                }
                if w < 0.0 {
                    return Err(CoreError::invalid_input("masses must be finite and >= 0.0"));
                }
            }
        }

        match source {
            CoordSource::Xyz { xs, ys, zs } => {
                for i in 0..n {
                    if !xs[i].is_finite() || !ys[i].is_finite() || !zs[i].is_finite() {
                        return Err(CoreError::non_finite(context));
                    }
                }
            }
            CoordSource::Flat { coords, .. } => {
                for &v in coords {
                    if !v.is_finite() {
                        return Err(CoreError::non_finite(context));
                    }
                }
            }
        }

        self.clear();
        if n == 0 {
            return Ok(());
        }

        self.coords_x.reserve(n);
        self.coords_y.reserve(n);
        self.coords_z.reserve(n);
        self.masses.reserve(n);

        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut min_z = f64::INFINITY;
        let mut max_z = f64::NEG_INFINITY;

        match source {
            CoordSource::Xyz { xs, ys, zs } => {
                for i in 0..n {
                    let x = xs[i].to_f64();
                    let y = ys[i].to_f64();
                    let z = zs[i].to_f64();
                    let m = masses.map_or(1.0, |ms| ms[i]);
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                    min_z = min_z.min(z);
                    max_z = max_z.max(z);
                    self.coords_x.push(x);
                    self.coords_y.push(y);
                    self.coords_z.push(z);
                    self.masses.push(m);
                }
            }
            CoordSource::Flat { coords, .. } => {
                for i in 0..n {
                    let base = i * 3;
                    let x = coords[base].to_f64();
                    let y = coords[base + 1].to_f64();
                    let z = coords[base + 2].to_f64();
                    let m = masses.map_or(1.0, |ms| ms[i]);
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                    min_z = min_z.min(z);
                    max_z = max_z.max(z);
                    self.coords_x.push(x);
                    self.coords_y.push(y);
                    self.coords_z.push(z);
                    self.masses.push(m);
                }
            }
        }

        self.leaf_of_point.resize(n, u32::MAX);
        self.next_in_leaf.resize(n, u32::MAX);

        let center = [
            0.5 * (min_x + max_x),
            0.5 * (min_y + max_y),
            0.5 * (min_z + max_z),
        ];
        // Use the largest axis span across all 3 axes so 1D-collinear and 2D-planar inputs
        // produce valid non-degenerate cubic octants.
        let span_max = ((max_x - min_x).max(max_y - min_y).max(max_z - min_z) * 1.01).max(1e-4);
        let half_span = 0.5 * span_max;

        self.nodes.push(ArenaNode3d::new(center, half_span));

        for i in 0..n {
            self.insert_iterative(
                i,
                self.coords_x[i],
                self.coords_y[i],
                self.coords_z[i],
                self.masses[i],
            );
        }
        Ok(())
    }

    fn insert_iterative(&mut self, point_idx: usize, x: f64, y: f64, z: f64, mass: f64) {
        let mut cur = 0usize;
        let mut depth = 0usize;

        loop {
            let node = self.nodes[cur];
            if node.point_count == 0 {
                let slot = &mut self.nodes[cur];
                slot.point_count = 1;
                slot.total_mass = mass;
                slot.com = [x, y, z];
                slot.leaf_head = point_idx as u32;
                self.leaf_of_point[point_idx] = cur as u32;
                self.next_in_leaf[point_idx] = u32::MAX;
                return;
            }

            Self::accumulate_com(&mut self.nodes[cur], x, y, z, mass);

            if node.is_leaf {
                let ref_idx = node.leaf_head as usize;
                let dx = self.coords_x[ref_idx] - x;
                let dy = self.coords_y[ref_idx] - y;
                let dz = self.coords_z[ref_idx] - z;
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if !dist_sq.is_finite()
                    || dist_sq < COINCIDENT_DIST_SQ
                    || node.is_unsplitable_at_depth(depth)
                {
                    self.next_in_leaf[point_idx] = self.nodes[cur].leaf_head;
                    self.nodes[cur].leaf_head = point_idx as u32;
                    self.leaf_of_point[point_idx] = cur as u32;
                    return;
                }

                // Subdivide leaf and re-insert its existing point chain into child octants.
                let head = self.nodes[cur].leaf_head;
                self.nodes[cur].leaf_head = u32::MAX;
                let base_child = self.subdivide_node(cur);

                let mut existing = head;
                while existing != u32::MAX {
                    let e_idx = existing as usize;
                    let next = self.next_in_leaf[e_idx];
                    self.next_in_leaf[e_idx] = u32::MAX;
                    let ex = self.coords_x[e_idx];
                    let ey = self.coords_y[e_idx];
                    let ez = self.coords_z[e_idx];
                    let em = self.masses[e_idx];
                    let oct = self.nodes[cur].octant(ex, ey, ez);
                    self.insert_into_subtree(base_child + oct, depth + 1, e_idx, ex, ey, ez, em);
                    existing = next;
                }

                let oct_new = self.nodes[cur].octant(x, y, z);
                cur = base_child + oct_new;
                depth += 1;
            } else {
                let oct = node.octant(x, y, z);
                cur = node.children[oct] as usize;
                depth += 1;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_into_subtree(
        &mut self,
        mut cur: usize,
        mut depth: usize,
        point_idx: usize,
        x: f64,
        y: f64,
        z: f64,
        mass: f64,
    ) {
        loop {
            let node = self.nodes[cur];
            if node.point_count == 0 {
                let slot = &mut self.nodes[cur];
                slot.point_count = 1;
                slot.total_mass = mass;
                slot.com = [x, y, z];
                slot.leaf_head = point_idx as u32;
                self.leaf_of_point[point_idx] = cur as u32;
                self.next_in_leaf[point_idx] = u32::MAX;
                return;
            }

            Self::accumulate_com(&mut self.nodes[cur], x, y, z, mass);

            if node.is_leaf {
                let ref_idx = node.leaf_head as usize;
                let dx = self.coords_x[ref_idx] - x;
                let dy = self.coords_y[ref_idx] - y;
                let dz = self.coords_z[ref_idx] - z;
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if !dist_sq.is_finite()
                    || dist_sq < COINCIDENT_DIST_SQ
                    || node.is_unsplitable_at_depth(depth)
                {
                    self.next_in_leaf[point_idx] = self.nodes[cur].leaf_head;
                    self.nodes[cur].leaf_head = point_idx as u32;
                    self.leaf_of_point[point_idx] = cur as u32;
                    return;
                }

                let head = self.nodes[cur].leaf_head;
                self.nodes[cur].leaf_head = u32::MAX;
                let base_child = self.subdivide_node(cur);

                let mut existing = head;
                while existing != u32::MAX {
                    let e_idx = existing as usize;
                    let next = self.next_in_leaf[e_idx];
                    self.next_in_leaf[e_idx] = u32::MAX;
                    let ex = self.coords_x[e_idx];
                    let ey = self.coords_y[e_idx];
                    let ez = self.coords_z[e_idx];
                    let em = self.masses[e_idx];
                    let oct = self.nodes[cur].octant(ex, ey, ez);
                    let child = base_child + oct;
                    let c_node = &mut self.nodes[child];
                    if c_node.point_count == 0 {
                        c_node.point_count = 1;
                        c_node.total_mass = em;
                        c_node.com = [ex, ey, ez];
                        c_node.leaf_head = e_idx as u32;
                    } else {
                        Self::accumulate_com(c_node, ex, ey, ez, em);
                        self.next_in_leaf[e_idx] = c_node.leaf_head;
                        c_node.leaf_head = e_idx as u32;
                    }
                    self.leaf_of_point[e_idx] = child as u32;
                    existing = next;
                }

                let oct_new = self.nodes[cur].octant(x, y, z);
                cur = base_child + oct_new;
                depth += 1;
            } else {
                let oct = node.octant(x, y, z);
                cur = node.children[oct] as usize;
                depth += 1;
            }
        }
    }

    #[inline]
    fn accumulate_com(slot: &mut ArenaNode3d, x: f64, y: f64, z: f64, mass: f64) {
        let old_c = f64::from(slot.point_count);
        let new_c = f64::from(slot.point_count + 1);
        let old_m = slot.total_mass;
        let new_m = old_m + mass;
        if new_m > 0.0 {
            slot.com[0] = (slot.com[0] * old_m + x * mass) / new_m;
            slot.com[1] = (slot.com[1] * old_m + y * mass) / new_m;
            slot.com[2] = (slot.com[2] * old_m + z * mass) / new_m;
        } else {
            slot.com[0] = (slot.com[0] * old_c + x) / new_c;
            slot.com[1] = (slot.com[1] * old_c + y) / new_c;
            slot.com[2] = (slot.com[2] * old_c + z) / new_c;
        }
        slot.total_mass = new_m;
        slot.point_count += 1;
    }

    fn subdivide_node(&mut self, parent_idx: usize) -> usize {
        let p = self.nodes[parent_idx];
        let quarter = p.half_span * 0.5;
        let base = self.nodes.len();
        self.nodes[parent_idx].is_leaf = false;

        for c in 0..8usize {
            let sx = if (c & 1) == 0 { -quarter } else { quarter };
            let sy = if (c & 2) == 0 { -quarter } else { quarter };
            let sz = if (c & 4) == 0 { -quarter } else { quarter };
            self.nodes[parent_idx].children[c] = (base + c) as u32;
            self.nodes.push(ArenaNode3d::new(
                [p.center[0] + sx, p.center[1] + sy, p.center[2] + sz],
                quarter,
            ));
        }
        base
    }
}
