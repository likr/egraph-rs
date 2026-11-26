//! Pure-Rust unit tests for neighborhood statistics helpers.

use egraph::neighborhood::KnnStats;

use super::types::{knn_stats_float_fields, knn_stats_int_fields};

#[test]
fn test_knn_stats_helpers() {
    let knn_stats = KnnStats {
        avg_deg1: 2.0,
        avg_deg2: 5.0,
        avg_row_k: 6.0,
        min_row_k: 4,
        max_row_k: 8,
        nnz: 60,
        n_1hop: 20,
        n_2hop: 30,
        n_truncated_1hop: 0,
        n_truncated_2hop: 0,
        n_filled_residual: 3,
        n_pure_2hop_rows: 7,
    };
    assert_eq!(knn_stats_float_fields(&knn_stats).len(), 3);
    assert_eq!(knn_stats_int_fields(&knn_stats)[2], ("nnz", 60));
}
