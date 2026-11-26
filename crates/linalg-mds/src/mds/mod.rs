//! Multidimensional Scaling (`pivot_mds`, `representative_mds`, `classical_mds`, `RdMds`),
//! Harel-Koren High-Dimensional Embedding (`hde`, `hde_nd`), Koren degree-normalized spectral
//! initialization, and isotropic 2D coordinate normalization.

mod classical;
mod hde;
mod hde_pca;
mod normalize;
mod pivot;
mod spectral;

pub use classical::{classical_mds, classical_mds_2d, classical_mds_from_distances};
pub use hde::{HdeConfig, hde, hde_nd, hde_nd_with_pivots, hde_with_pivots};
pub use normalize::{normalize_layout_2d, robust_normalize_layout_2d};
pub use pivot::{
    PivotMdsConfig, pivot_mds, pivot_mds_from_distances, pivot_mds_with_pivots, representative_mds,
};
pub use spectral::{RdMds, koren_spectral_init, rd_mds};

#[cfg(test)]
mod tests_hde_edge_cases;
#[cfg(test)]
mod tests_hde_oracle;
#[cfg(test)]
mod tests_hde_scale;
#[cfg(test)]
mod tests_mds;
