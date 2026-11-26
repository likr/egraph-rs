//! R1/R2 regularized DoG Varadhan landmark tables, R3 Horvitz-Thompson repulsion scaling,
//! coarse diffusive scaffold tables, landmark ALT pair distances, and relative distance floors.

mod config;
mod dog_landmarks;
mod floors;
mod rhgh_landmarks;

#[cfg(test)]
mod tests;

use egraph_core::{CoreError, CoreResult};

pub use config::{DogVaradhanConfig, DogVaradhanStats, RhghLandmarkConfig, RhghLandmarkStats};
pub use dog_landmarks::{
    compute_regularized_dog_varadhan_landmarks,
    compute_regularized_dog_varadhan_landmarks_from_matrix,
};
pub use floors::{apply_relative_distance_floor, compute_horvitz_thompson_gamma};
pub use rhgh_landmarks::compute_pivoted_rhgh_landmarks;

/// Precomputed row-major `(num_nodes, num_landmarks)` R1/R2 DoG Varadhan landmark tables
/// (plus optional coarse diffusive scaffold table) supporting fast ALT lower-bound distance queries.
#[derive(Debug, Clone, PartialEq)]
pub struct DogVaradhanLandmarks {
    pub num_nodes: usize,
    pub landmarks: Vec<u32>,
    pub d_dog: Vec<f64>,
    pub d_coarse: Option<Vec<f64>>,
}

impl DogVaradhanLandmarks {
    /// Returns the number of vertices.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of landmarks.
    #[inline]
    pub fn num_landmarks(&self) -> usize {
        self.landmarks.len()
    }

    /// Returns the selected landmark vertex IDs.
    #[inline]
    pub fn landmarks(&self) -> &[u32] {
        &self.landmarks
    }

    /// Returns the row-major `(num_nodes, num_landmarks)` DoG Varadhan table.
    #[inline]
    pub fn d_dog(&self) -> &[f64] {
        &self.d_dog
    }

    /// Returns the optional coarse diffusive Varadhan scaffold table.
    #[inline]
    pub fn d_coarse(&self) -> Option<&[f64]> {
        self.d_coarse.as_deref()
    }

    /// Evaluates landmark ALT lower-bound distances
    /// $\max_p |D_{\mathrm{dog}}[u, p] - D_{\mathrm{dog}}[v, p]|$ (and max over `d_coarse` when present)
    /// for all pairs `(u[i], v[i])`.
    pub fn alt_pair_distances(&self, u: &[u32], v: &[u32]) -> CoreResult<Vec<f64>> {
        if u.len() != v.len() {
            return Err(CoreError::dimension_mismatch(
                u.len(),
                v.len(),
                "DogVaradhanLandmarks::alt_pair_distances",
            ));
        }
        let k_lm = self.landmarks.len();
        let expected = self.num_nodes.saturating_mul(k_lm);
        if self.d_dog.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                self.d_dog.len(),
                "DogVaradhanLandmarks::d_dog",
            ));
        }
        if let Some(dc) = &self.d_coarse
            && dc.len() != expected
        {
            return Err(CoreError::dimension_mismatch(
                expected,
                dc.len(),
                "DogVaradhanLandmarks::d_coarse",
            ));
        }

        let mut out = Vec::with_capacity(u.len());
        for (&ui, &vi) in u.iter().zip(v.iter()) {
            let u_idx = ui as usize;
            let v_idx = vi as usize;
            if u_idx >= self.num_nodes {
                return Err(CoreError::index_out_of_bounds(
                    u_idx,
                    self.num_nodes,
                    "DogVaradhanLandmarks::alt_pair_distances::u",
                ));
            }
            if v_idx >= self.num_nodes {
                return Err(CoreError::index_out_of_bounds(
                    v_idx,
                    self.num_nodes,
                    "DogVaradhanLandmarks::alt_pair_distances::v",
                ));
            }
            if k_lm == 0 || self.num_nodes <= 1 {
                out.push(0.0);
                continue;
            }
            let ru = &self.d_dog[u_idx * k_lm..(u_idx + 1) * k_lm];
            let rv = &self.d_dog[v_idx * k_lm..(v_idx + 1) * k_lm];
            let mut best = 0.0_f64;
            for (&a, &b) in ru.iter().zip(rv.iter()) {
                let diff = (a - b).abs();
                if diff > best {
                    best = diff;
                }
            }
            if let Some(dc) = &self.d_coarse {
                let cu = &dc[u_idx * k_lm..(u_idx + 1) * k_lm];
                let cv = &dc[v_idx * k_lm..(v_idx + 1) * k_lm];
                for (&a, &b) in cu.iter().zip(cv.iter()) {
                    let diff = (a - b).abs();
                    if diff > best {
                        best = diff;
                    }
                }
            }
            out.push(best);
        }
        Ok(out)
    }
}
