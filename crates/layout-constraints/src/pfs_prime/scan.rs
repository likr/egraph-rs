use egraph_core::Scalar;

use super::solver::PfsPrimeOverlapRemoval;

impl<S: Scalar> PfsPrimeOverlapRemoval<S> {
    fn compute_min_positive_gap(&mut self) -> Option<f64> {
        let mut min_gap: Option<f64> = None;

        for coords in [&self.work_x, &self.work_y] {
            self.gap_scratch.clear();
            self.gap_scratch.extend_from_slice(coords);
            self.gap_scratch.sort_unstable_by(f64::total_cmp);
            for w in self.gap_scratch.windows(2) {
                if w[1] > w[0] {
                    let gap = w[1] - w[0];
                    if gap > 0.0 {
                        min_gap = Some(min_gap.map_or(gap, |m| m.min(gap)));
                    }
                }
            }
        }

        min_gap
    }

    pub(super) fn jitter_coincident_points(&mut self) {
        let n = self.num_nodes;
        if n < 2 {
            return;
        }
        let min_pos_gap = self.compute_min_positive_gap();

        self.sort_pts.clear();
        for i in 0..n {
            self.sort_pts
                .push((self.work_x[i], self.work_y[i], i as u32));
        }
        self.sort_pts.sort_unstable_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then_with(|| a.1.total_cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });

        let mut i = 0usize;
        while i < n {
            let (base_x, base_y, _) = self.sort_pts[i];
            let mut j = i + 1;
            while j < n {
                let (nx, ny, _) = self.sort_pts[j];
                if nx == base_x && ny == base_y {
                    j += 1;
                } else {
                    break;
                }
            }
            if j - i > 1 {
                let coord_ulp = base_x.abs().max(base_y.abs()) * S::EPSILON.to_f64();
                for k in (i + 1)..j {
                    let idx = self.sort_pts[k].2 as usize;
                    let extent = self.eff_hw_req[idx].max(self.eff_hh_req[idx]).max(1e-2);
                    let desired_scale = (extent * 0.05).max(4.0 * coord_ulp).max(1e-5);
                    let jitter_scale = match min_pos_gap {
                        Some(gap) => desired_scale.min(0.2375 * gap),
                        None => desired_scale,
                    };
                    let angle = self.rng.next_f64() * std::f64::consts::TAU;
                    let radius = jitter_scale * (0.25 + 0.75 * self.rng.next_f64());
                    let (sin_a, cos_a) = angle.sin_cos();
                    self.work_x[idx] += radius * cos_a;
                    self.work_y[idx] += radius * sin_a;
                }
            }
            i = j;
        }
    }

    #[inline]
    fn pfsp_force(
        &self,
        i: usize,
        j: usize,
        sep_eps: f64,
        slack_x: f64,
        slack_y: f64,
    ) -> (f64, f64) {
        let base_req_x = self.eff_hw_req[i] + self.eff_hw_req[j];
        let base_req_y = self.eff_hh_req[i] + self.eff_hh_req[j];
        if base_req_x <= 0.0 || base_req_y <= 0.0 {
            return (0.0, 0.0);
        }
        let req_x = base_req_x * sep_eps + slack_x;
        let req_y = base_req_y * sep_eps + slack_y;

        let dx = self.work_x[j] - self.work_x[i];
        let dy = self.work_y[j] - self.work_y[i];
        if dx == 0.0 && dy == 0.0 {
            return (0.0, 0.0);
        }
        let adx = dx.abs();
        let ady = dy.abs();

        if dx == 0.0 {
            // Vertical alignment: |g_ij| = inf > G_ij -> pure vertical separation.
            let fy = dy.signum() * (req_y - ady);
            return (0.0, fy);
        }

        let g = dy / dx;
        let big_g = req_y / req_x;
        if g.abs() <= big_g {
            let fx = dx.signum() * (req_x - adx);
            (fx, fx * g)
        } else {
            let fy = dy.signum() * (req_y - ady);
            (fy / g, fy)
        }
    }

    pub(super) fn horizontal_scan(&mut self, sep_eps: f64, slack_x: f64, slack_y: f64) {
        let n = self.num_nodes;
        for (idx, item) in self.order.iter_mut().enumerate() {
            *item = idx;
        }
        self.order.sort_unstable_by(|&a, &b| {
            self.work_x[a]
                .total_cmp(&self.work_x[b])
                .then_with(|| a.cmp(&b))
        });

        let mut i = 0usize;
        let mut sigma = 0.0_f64;
        let mut lmin = self.order[0];

        while i < n {
            let x_i = self.work_x[self.order[i]];
            let mut k = i;
            while k + 1 < n && (self.work_x[self.order[k + 1]] - x_i).abs() == 0.0 {
                k += 1;
            }

            let mut gamma = 0.0_f64;
            if x_i > self.work_x[lmin] {
                for m in i..=k {
                    let v = self.order[m];
                    let mut gamma_pp = 0.0_f64;
                    for j in 0..i {
                        let node_j = self.order[j];
                        let (fx, _) = self.pfsp_force(node_j, v, sep_eps, slack_x, slack_y);
                        gamma_pp = gamma_pp.max(self.up_gamma_x[node_j] + fx);
                    }
                    let gamma_p = if (self.work_x[v] - self.work_x[lmin]) + gamma_pp
                        < (self.eff_hw_req[v] - self.eff_hw_req[lmin])
                    {
                        sigma
                    } else {
                        gamma_pp
                    };
                    gamma = gamma.max(gamma_p);
                }
            }

            for m in i..=k {
                let r = self.order[m];
                self.up_gamma_x[r] = gamma;
                if (self.work_x[r] - self.work_x[lmin])
                    < (self.eff_hw_req[r] - self.eff_hw_req[lmin])
                {
                    lmin = r;
                }
            }

            let mut delta = 0.0_f64;
            for m in i..=k {
                let u = self.order[m];
                for j in (k + 1)..n {
                    let v = self.order[j];
                    let (fx, _) = self.pfsp_force(u, v, sep_eps, slack_x, slack_y);
                    delta = delta.max(fx);
                }
            }
            sigma += delta;
            i = k + 1;
        }
    }

    pub(super) fn vertical_scan(&mut self, sep_eps: f64, slack_x: f64, slack_y: f64) {
        let n = self.num_nodes;
        for (idx, item) in self.order.iter_mut().enumerate() {
            *item = idx;
        }
        self.order.sort_unstable_by(|&a, &b| {
            self.work_y[a]
                .total_cmp(&self.work_y[b])
                .then_with(|| a.cmp(&b))
        });

        let mut i = 0usize;
        let mut sigma = 0.0_f64;
        let mut lmin = self.order[0];

        while i < n {
            let y_i = self.work_y[self.order[i]];
            let mut k = i;
            while k + 1 < n && (self.work_y[self.order[k + 1]] - y_i).abs() == 0.0 {
                k += 1;
            }

            let mut gamma = 0.0_f64;
            if y_i > self.work_y[lmin] {
                for m in i..=k {
                    let v = self.order[m];
                    let mut gamma_pp = 0.0_f64;
                    for j in 0..i {
                        let node_j = self.order[j];
                        let (_, fy) = self.pfsp_force(node_j, v, sep_eps, slack_x, slack_y);
                        gamma_pp = gamma_pp.max(self.up_gamma_y[node_j] + fy);
                    }
                    let gamma_p = if (self.work_y[v] - self.work_y[lmin]) + gamma_pp
                        < (self.eff_hh_req[v] - self.eff_hh_req[lmin])
                    {
                        sigma
                    } else {
                        gamma_pp
                    };
                    gamma = gamma.max(gamma_p);
                }
            }

            for m in i..=k {
                let r = self.order[m];
                self.up_gamma_y[r] = gamma;
                if (self.work_y[r] - self.work_y[lmin])
                    < (self.eff_hh_req[r] - self.eff_hh_req[lmin])
                {
                    lmin = r;
                }
            }

            let mut delta = 0.0_f64;
            for m in i..=k {
                let u = self.order[m];
                for j in (k + 1)..n {
                    let v = self.order[j];
                    let (_, fy) = self.pfsp_force(u, v, sep_eps, slack_x, slack_y);
                    delta = delta.max(fy);
                }
            }
            sigma += delta;
            i = k + 1;
        }
    }
}
