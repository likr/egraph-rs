//! Multidimensional Scaling ([`ClassicalMds`], [`PivotMds`], [`RdMds`]), Harel-Koren
//! High-Dimensional Embedding ([`Hde`]), Koren degree-normalized spectral initialization
//! ([`KorenSpectral`]), and isotropic Euclidean coordinate normalization.

mod classical;
mod hde;
mod hde_pca;
mod normalize;
mod normalize_robust;
mod pivot;
mod pivot_kernel;
mod rd_mds;
mod spectral;
mod storage;

pub use classical::{ClassicalMds, ClassicalMdsConfig};
pub use hde::{Hde, HdeConfig};
pub use normalize::{normalize_euclidean_layout, normalize_layout_2d};
pub use normalize_robust::{robust_normalize_euclidean_layout, robust_normalize_layout_2d};
pub use pivot::{PivotMds, PivotMdsConfig};
pub use rd_mds::{RdMds, RdMdsConfig};
pub use spectral::{KorenSpectral, KorenSpectralConfig};

#[cfg(test)]
mod tests_generators;
#[cfg(test)]
mod tests_hde_edge_cases;
#[cfg(test)]
mod tests_hde_oracle;
#[cfg(test)]
mod tests_hde_scale;
#[cfg(test)]
mod tests_mds;
