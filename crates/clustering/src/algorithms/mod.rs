pub mod infomap;
pub mod label_propagation;
pub mod louvain;
pub mod spectral;

// Re-export all algorithms
pub use infomap::*;
pub use label_propagation::*;
pub use louvain::*;
pub use spectral::*;
