//! Classical, Pivot, Landmark, Harel-Koren HDE, and Koren spectral multidimensional scaling for `egraph-rs`.

#![warn(missing_docs)]

mod mds;

pub use mds::{
    ClassicalMds, ClassicalMdsConfig, Hde, HdeConfig, KorenSpectral, KorenSpectralConfig, PivotMds,
    PivotMdsConfig, RdMds, RdMdsConfig, normalize_euclidean_layout, normalize_layout_2d,
    robust_normalize_euclidean_layout, robust_normalize_layout_2d,
};
