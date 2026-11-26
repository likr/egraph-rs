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
pub use rhgh_landmarks::{compute_pivoted_rhgh_landmarks, compute_pivoted_rhgh_landmarks_result};

fn eval_alt_pair_distances(
    num_nodes: usize,
    landmarks: &[u32],
    d_primary: &[f64],
    d_coarse: Option<&[f64]>,
    u: &[u32],
    v: &[u32],
    contexts: [&'static str; 4],
) -> CoreResult<Vec<f64>> {
    let [ctx_primary, ctx_pairs, ctx_u, ctx_v] = contexts;
    if u.len() != v.len() {
        return Err(CoreError::dimension_mismatch(u.len(), v.len(), ctx_pairs));
    }
    let k_lm = landmarks.len();
    let expected = num_nodes.saturating_mul(k_lm);
    if d_primary.len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            d_primary.len(),
            ctx_primary,
        ));
    }
    if let Some(dc) = d_coarse
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
        if u_idx >= num_nodes {
            return Err(CoreError::index_out_of_bounds(u_idx, num_nodes, ctx_u));
        }
        if v_idx >= num_nodes {
            return Err(CoreError::index_out_of_bounds(v_idx, num_nodes, ctx_v));
        }
        if k_lm == 0 || num_nodes <= 1 {
            out.push(0.0);
            continue;
        }
        let ru = &d_primary[u_idx * k_lm..(u_idx + 1) * k_lm];
        let rv = &d_primary[v_idx * k_lm..(v_idx + 1) * k_lm];
        let mut best = 0.0_f64;
        for (&a, &b) in ru.iter().zip(rv.iter()) {
            let diff = (a - b).abs();
            if diff > best {
                best = diff;
            }
        }
        if let Some(dc) = d_coarse {
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
        eval_alt_pair_distances(
            self.num_nodes,
            &self.landmarks,
            &self.d_dog,
            self.d_coarse.as_deref(),
            u,
            v,
            [
                "DogVaradhanLandmarks::d_dog",
                "DogVaradhanLandmarks::alt_pair_distances",
                "DogVaradhanLandmarks::alt_pair_distances::u",
                "DogVaradhanLandmarks::alt_pair_distances::v",
            ],
        )
    }
}

/// Precomputed row-major `(num_nodes, num_landmarks)` pivoted `RHGH` landmark distance table
/// supporting fast ALT lower-bound distance queries and conversion to [`DogVaradhanLandmarks`].
#[derive(Debug, Clone, PartialEq)]
pub struct RhghLandmarks {
    pub num_nodes: usize,
    pub landmarks: Vec<u32>,
    pub d_rhgh: Vec<f64>,
}

impl RhghLandmarks {
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

    /// Returns the row-major `(num_nodes, num_landmarks)` pivoted `RHGH` distance table.
    #[inline]
    pub fn d_rhgh(&self) -> &[f64] {
        &self.d_rhgh
    }

    /// Evaluates landmark ALT lower-bound distances
    /// $\max_p |D_{\mathrm{rhgh}}[u, p] - D_{\mathrm{rhgh}}[v, p]|$ for all pairs `(u[i], v[i])`.
    pub fn alt_pair_distances(&self, u: &[u32], v: &[u32]) -> CoreResult<Vec<f64>> {
        eval_alt_pair_distances(
            self.num_nodes,
            &self.landmarks,
            &self.d_rhgh,
            None,
            u,
            v,
            [
                "RhghLandmarks::d_rhgh",
                "RhghLandmarks::alt_pair_distances",
                "RhghLandmarks::alt_pair_distances::u",
                "RhghLandmarks::alt_pair_distances::v",
            ],
        )
    }

    /// Converts this `RhghLandmarks` table into [`DogVaradhanLandmarks`] without copying.
    #[inline]
    pub fn into_dog_landmarks(self) -> DogVaradhanLandmarks {
        DogVaradhanLandmarks {
            num_nodes: self.num_nodes,
            landmarks: self.landmarks,
            d_dog: self.d_rhgh,
            d_coarse: None,
        }
    }

    /// Clones this `RhghLandmarks` table into [`DogVaradhanLandmarks`].
    #[inline]
    pub fn to_dog_landmarks(&self) -> DogVaradhanLandmarks {
        self.clone().into_dog_landmarks()
    }
}

impl From<RhghLandmarks> for DogVaradhanLandmarks {
    #[inline]
    fn from(value: RhghLandmarks) -> Self {
        value.into_dog_landmarks()
    }
}
