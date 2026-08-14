use num_traits::Float;
use petgraph_drawing::DrawingValue;

/// Node in a 2D Quadtree for Barnes-Hut spatial force approximation.
#[derive(Debug, Clone)]
pub struct QuadtreeNode<S> {
    pub center_x: S,
    pub center_y: S,
    pub width: S,
    pub height: S,
    pub count: usize,
    pub center_of_mass_x: S,
    pub center_of_mass_y: S,
    pub point_index: Option<usize>,
    pub children: Option<Box<[QuadtreeNode<S>; 4]>>, // [NW, NE, SW, SE]
}

impl<S> QuadtreeNode<S>
where
    S: DrawingValue + Float + Default,
{
    pub fn new(center_x: S, center_y: S, width: S, height: S) -> Self {
        Self {
            center_x,
            center_y,
            width,
            height,
            count: 0,
            center_of_mass_x: S::zero(),
            center_of_mass_y: S::zero(),
            point_index: None,
            children: None,
        }
    }

    /// Inserts a point into the quadtree node.
    pub fn insert(&mut self, point_idx: usize, x: S, y: S, points: &[[S; 2]]) {
        if self.count == 0 {
            self.point_index = Some(point_idx);
            self.count = 1;
            self.center_of_mass_x = x;
            self.center_of_mass_y = y;
            return;
        }

        // Update center of mass incrementally
        let old_count_s = S::from_usize(self.count).unwrap();
        let new_count_s = S::from_usize(self.count + 1).unwrap();
        self.center_of_mass_x = (self.center_of_mass_x * old_count_s + x) / new_count_s;
        self.center_of_mass_y = (self.center_of_mass_y * old_count_s + y) / new_count_s;
        self.count += 1;

        // If this is a leaf node containing a single point, subdivide into 4 quadrants
        if self.children.is_none() {
            if let Some(existing_idx) = self.point_index.take() {
                // If existing point is almost identical in coordinates to new point, avoid infinite recursion
                let ex = points[existing_idx][0];
                let ey = points[existing_idx][1];
                let dist_sq = (ex - x) * (ex - x) + (ey - y) * (ey - y);
                if dist_sq < S::from_f32(1e-12).unwrap() {
                    // Points overlap, keep both count but do not subdivide infinitely
                    self.point_index = Some(existing_idx);
                    return;
                }

                self.subdivide();
                let child_idx = self.child_quadrant(ex, ey);
                if let Some(ref mut children) = self.children {
                    children[child_idx].insert(existing_idx, ex, ey, points);
                }
            } else {
                self.subdivide();
            }
        }

        let child_idx = self.child_quadrant(x, y);
        if let Some(ref mut children) = self.children {
            children[child_idx].insert(point_idx, x, y, points);
        }
    }

    fn subdivide(&mut self) {
        let half_w = self.width / S::from_f32(2.0).unwrap();
        let half_h = self.height / S::from_f32(2.0).unwrap();
        let quarter_w = self.width / S::from_f32(4.0).unwrap();
        let quarter_h = self.height / S::from_f32(4.0).unwrap();

        // 0: NW, 1: NE, 2: SW, 3: SE
        self.children = Some(Box::new([
            QuadtreeNode::new(
                self.center_x - quarter_w,
                self.center_y - quarter_h,
                half_w,
                half_h,
            ),
            QuadtreeNode::new(
                self.center_x + quarter_w,
                self.center_y - quarter_h,
                half_w,
                half_h,
            ),
            QuadtreeNode::new(
                self.center_x - quarter_w,
                self.center_y + quarter_h,
                half_w,
                half_h,
            ),
            QuadtreeNode::new(
                self.center_x + quarter_w,
                self.center_y + quarter_h,
                half_w,
                half_h,
            ),
        ]));
    }

    #[inline]
    fn child_quadrant(&self, x: S, y: S) -> usize {
        let right = if x >= self.center_x { 1 } else { 0 };
        let bottom = if y >= self.center_y { 2 } else { 0 };
        right + bottom
    }

    /// Evaluates Barnes-Hut repulsive forces for KL divergence.
    ///
    /// Computes accumulated unnormalized Cauchy weight sum `z_sum` and force vector `(f_x, f_y)` for node `point_idx`.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_kl_repulsion(
        &self,
        point_idx: usize,
        x: S,
        y: S,
        theta_sq: S,
        z_sum: &mut S,
        f_x: &mut S,
        f_y: &mut S,
    ) {
        if self.count == 0 {
            return;
        }

        let dx = x - self.center_of_mass_x;
        let dy = y - self.center_of_mass_y;
        let dist_sq = dx * dx + dy * dy;

        // If leaf with single point
        if self.children.is_none() {
            if let Some(other_idx) = self.point_index {
                if other_idx != point_idx {
                    let w = S::one() / (S::one() + dist_sq);
                    let count_s = S::from_usize(self.count).unwrap();
                    *z_sum += count_s * w;
                    let mult = count_s * w * w;
                    *f_x += mult * dx;
                    *f_y += mult * dy;
                }
            }
            return;
        }

        // Barnes-Hut opening criterion: (max(width, height))^2 / dist_sq < theta^2
        let max_dim = self.width.max(self.height);
        let cell_size_sq = max_dim * max_dim;

        if cell_size_sq < theta_sq * dist_sq {
            // Treat cell as a summary point at its center of mass
            let w = S::one() / (S::one() + dist_sq);
            let count_s = S::from_usize(self.count).unwrap();
            *z_sum += count_s * w;
            let mult = count_s * w * w;
            *f_x += mult * dx;
            *f_y += mult * dy;
        } else if let Some(ref children) = self.children {
            for child in children.iter() {
                child.compute_kl_repulsion(point_idx, x, y, theta_sq, z_sum, f_x, f_y);
            }
        }
    }

    /// Evaluates Barnes-Hut repulsive forces for the entropy term C_ENT.
    ///
    /// Computes gradient force vector `(f_x, f_y)` based on:
    /// `sum_j |M| * (y_i - c_M) / (||y_i - c_M|| * (||y_i - c_M|| + epsilon_r))`
    #[allow(clippy::too_many_arguments)]
    pub fn compute_entropy_repulsion(
        &self,
        point_idx: usize,
        x: S,
        y: S,
        theta_sq: S,
        epsilon_r: S,
        f_x: &mut S,
        f_y: &mut S,
    ) {
        if self.count == 0 {
            return;
        }

        let dx = x - self.center_of_mass_x;
        let dy = y - self.center_of_mass_y;
        let dist_sq = dx * dx + dy * dy;

        // If leaf with single point
        if self.children.is_none() {
            if let Some(other_idx) = self.point_index {
                if other_idx != point_idx {
                    let dist = dist_sq.sqrt();
                    if dist > S::from_f32(1e-6).unwrap() {
                        let count_s = S::from_usize(self.count).unwrap();
                        let mult = count_s / (dist * (dist + epsilon_r));
                        *f_x += mult * dx;
                        *f_y += mult * dy;
                    }
                }
            }
            return;
        }

        // Barnes-Hut opening criterion
        let max_dim = self.width.max(self.height);
        let cell_size_sq = max_dim * max_dim;

        if cell_size_sq < theta_sq * dist_sq {
            let dist = dist_sq.sqrt();
            if dist > S::from_f32(1e-6).unwrap() {
                let count_s = S::from_usize(self.count).unwrap();
                let mult = count_s / (dist * (dist + epsilon_r));
                *f_x += mult * dx;
                *f_y += mult * dy;
            }
        } else if let Some(ref children) = self.children {
            for child in children.iter() {
                child.compute_entropy_repulsion(point_idx, x, y, theta_sq, epsilon_r, f_x, f_y);
            }
        }
    }
}

/// 2D Quadtree for accelerating N-body graph layout force calculations.
#[derive(Debug, Clone)]
pub struct Quadtree<S> {
    pub root: QuadtreeNode<S>,
}

impl<S> Quadtree<S>
where
    S: DrawingValue + Float + Default,
{
    /// Builds a Quadtree enclosing all given 2D coordinates.
    pub fn build(points: &[[S; 2]]) -> Self {
        let n = points.len();
        if n == 0 {
            return Self {
                root: QuadtreeNode::new(S::zero(), S::zero(), S::one(), S::one()),
            };
        }

        let mut min_x = points[0][0];
        let mut max_x = points[0][0];
        let mut min_y = points[0][1];
        let mut max_y = points[0][1];

        for p in points.iter().skip(1) {
            min_x = min_x.min(p[0]);
            max_x = max_x.max(p[0]);
            min_y = min_y.min(p[1]);
            max_y = max_y.max(p[1]);
        }

        let center_x = (min_x + max_x) / S::from_f32(2.0).unwrap();
        let center_y = (min_y + max_y) / S::from_f32(2.0).unwrap();
        let width = ((max_x - min_x) * S::from_f32(1.01).unwrap()).max(S::from_f32(1e-4).unwrap());
        let height = ((max_y - min_y) * S::from_f32(1.01).unwrap()).max(S::from_f32(1e-4).unwrap());

        let mut root = QuadtreeNode::new(center_x, center_y, width, height);

        for (idx, p) in points.iter().enumerate() {
            root.insert(idx, p[0], p[1], points);
        }

        Self { root }
    }
}
