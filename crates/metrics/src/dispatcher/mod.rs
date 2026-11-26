//! Target-selective metric evaluation dispatcher avoiding unnecessary computation.

mod config;
mod evaluate;
mod report;

#[cfg(test)]
mod tests;

#[cfg(feature = "eval-baselines")]
pub use config::ClusterBaselineConfig;
pub use config::MetricConfig;
pub use evaluate::{
    evaluate_metrics, evaluate_metrics_with_communities, evaluate_metrics_with_community_weights,
};
pub use report::{MetricReport, MetricTarget};
