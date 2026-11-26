//! # tsNET, BH-tsNET, FIt-tsNET & L-tsNET Graph Layout Algorithms
//!
//! This crate provides the `TsNet`, `BhTsNet`, `FitTsNet`, and `LTsNet` layout algorithms:
//! - `TsNet`: Full-matrix t-SNE-based graph layout optimizing KL divergence, early compression, and entropy.
//! - `BhTsNet`: Accelerated O(N log N) Barnes-Hut tsNET combining Partial BFS and Quadtree spatial approximations.
//! - `FitTsNet`: Accelerated O(N log N) Fast Interpolation tsNET combining Partial BFS, 2D FFT interpolation for KL divergence, and Quadtree for entropy.
//! - `LTsNet`: Linear-time O(N) L-tsNET extending 2D FFT interpolation to both KL divergence and entropy gradients.
//!
//! ## References
//!
//! - Kruiger, J. F., Rauber, P. E., Martins, R. M., Kerren, A., Kobourov, S., & Telea, A. C. (2017).
//!   Graph Layouts by t-SNE. *Computer Graphics Forum*, 36(3), 283-294.
//! - Meidiana, A., Hong, S.-H., & Ma, K.-L. (2025).
//!   BH-tsNET, FIt-tsNET, L-tsNET: Fast tsNET Algorithms for Large Graph Drawing. *arXiv:2509.19785*.

pub mod bh_ts_net;
pub mod bh_ts_net_builder;
pub mod fft;
pub mod fit_interpolation;
pub mod fit_ts_net;
pub mod fit_ts_net_builder;
pub mod l_ts_net;
pub mod l_ts_net_builder;
pub mod partial_bfs;
pub mod quadtree;
mod ts_net;
mod ts_net_builder;

pub use bh_ts_net::BhTsNet;
pub use bh_ts_net_builder::BhTsNetBuilder;
pub use fit_ts_net::FitTsNet;
pub use fit_ts_net_builder::FitTsNetBuilder;
pub use l_ts_net::LTsNet;
pub use l_ts_net_builder::LTsNetBuilder;
pub use partial_bfs::{
    compute_knn_probabilities, compute_knn_probabilities_from_csr,
    compute_knn_probabilities_with_params, compute_knn_probabilities_with_perplexities,
    compute_partial_bfs_probabilities, knn_from_csr, SparseP,
};
pub use ts_net::TsNet;
pub use ts_net_builder::TsNetBuilder;
