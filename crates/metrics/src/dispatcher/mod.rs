//! Target-selective metric evaluation dispatcher avoiding unnecessary computation.

#[cfg(feature = "eval-baselines")]
mod baselines;
mod config;
mod evaluate;
mod inputs;
mod report;
mod spatial;

#[cfg(test)]
mod tests;

pub use config::{ClusterBaselineConfig, MetricConfig};
pub use evaluate::{
    evaluate_metrics, evaluate_metrics_with_communities, evaluate_metrics_with_community_weights,
    evaluate_metrics_with_inputs,
};
pub use inputs::MetricInputs;
pub use report::{ClusterBaselineReport, MetricReport, MetricTarget};
