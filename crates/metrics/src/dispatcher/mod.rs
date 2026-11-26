//! Target-selective metric evaluation dispatcher avoiding unnecessary computation.

#[cfg(feature = "eval-baselines")]
mod baselines;
#[cfg(feature = "eval-baselines")]
mod baselines_manifold;
#[cfg(feature = "eval-baselines")]
mod baselines_nd;
mod config;
mod evaluate;
mod evaluate_manifold;
mod evaluate_nd;
mod report;

#[cfg(test)]
mod tests;

#[cfg(feature = "eval-baselines")]
pub use config::ClusterBaselineConfig;
pub use config::{MetricConfig, MetricInputs};
pub use evaluate::{
    MetricLayout, evaluate_metrics, evaluate_metrics_with_communities,
    evaluate_metrics_with_community_weights, evaluate_metrics_with_inputs,
};
pub use evaluate_nd::{
    evaluate_metrics_nd, evaluate_metrics_nd_with_communities,
    evaluate_metrics_nd_with_community_weights, evaluate_metrics_nd_with_inputs,
};
pub use report::{MetricReport, MetricTarget};
