//! # Stochastic Gradient Descent (SGD) Layout Algorithms
//!
//! This crate provides implementations of various Stochastic Gradient Descent (SGD) based
//! graph layout algorithms. SGD layout is a force-directed graph drawing technique that
//! positions nodes to minimize a stress function by following the gradient of the stress
//! with a series of small adjustment steps.
//!
//! ## Features
//!
//! - Multiple SGD variant implementations (Full, Sparse, Distance-Adjusted)
//! - Various learning rate schedulers with different decay patterns
//! - Customizable distance and weight functions
//!
//! ## Algorithm
//!
//! SGD layout works by iteratively moving nodes to better positions by minimizing the difference
//! between the graph-theoretical distances and the geometric distances in the layout. The algorithm
//! uses a learning rate parameter that typically decreases over time according to a schedule.

mod full_sgd;
mod scheduler;
mod sgd;
mod sparse_sgd;

pub use full_sgd::FullSgd;
pub use scheduler::*;
pub use sgd::Sgd;
pub use sparse_sgd::SparseSgd;
