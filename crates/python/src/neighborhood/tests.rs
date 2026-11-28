//! Pure-Rust unit tests for neighborhood statistics helpers.

use egraph::neighborhood::KnnStats;

use super::types::{knn_stats_float_fields, knn_stats_int_fields};

#[test]
fn test_knn_stats_helpers() {
    let mut knn_stats = KnnStats::default();
    knn_stats.avg_deg1 = 2.0;
    knn_stats.avg_deg2 = 5.0;
    knn_stats.avg_row_k = 6.0;
    knn_stats.min_row_k = 4;
    knn_stats.max_row_k = 8;
    knn_stats.nnz = 60;
    knn_stats.n_1hop = 20;
    knn_stats.n_2hop = 30;
    knn_stats.n_truncated_1hop = 0;
    knn_stats.n_truncated_2hop = 0;
    knn_stats.n_filled_residual = 3;
    knn_stats.n_pure_2hop_rows = 7;
    assert_eq!(knn_stats_float_fields(&knn_stats).len(), 3);
    assert_eq!(knn_stats_int_fields(&knn_stats)[2], ("nnz", 60));
}
