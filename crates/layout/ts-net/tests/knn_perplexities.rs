use petgraph_layout_ts_net::{
    compute_knn_probabilities_from_csr, compute_knn_probabilities_with_params,
    compute_knn_probabilities_with_perplexities, knn_from_csr, SparseP,
};

fn assert_sparse_p_bit_equal(actual: &SparseP<f64>, expected: &SparseP<f64>, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: row count mismatch");
    for (i, (row_a, row_e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(row_a.len(), row_e.len(), "{label}: row {i} len mismatch");
        for (&(ja, pa), &(je, pe)) in row_a.iter().zip(row_e.iter()) {
            assert_eq!(ja, je, "{label}: row {i} col mismatch");
            assert_eq!(
                pa.to_bits(),
                pe.to_bits(),
                "{label}: row {i} col {ja} value mismatch: {pa} vs {pe}"
            );
        }
    }
}

#[test]
fn test_compute_knn_probabilities_with_perplexities_matches_scalar_uniform() {
    let knn: Vec<Vec<(usize, f64)>> = vec![
        vec![(0, 0.0), (1, 3.0), (1, 0.4), (2, 1.2), (3, 2.5), (4, 4.0)],
        vec![(0, 0.4), (2, 0.8), (3, 1.5)],
        vec![(1, 0.8), (3, 1.1)],
        vec![(2, 1.1)],
        vec![],
        vec![(4, 0.5), (3, 1.0), (1, 2.2), (0, 3.5), (2, 5.0)],
    ];
    let n = knn.len();
    let perp = 2.25f64;
    let expected = compute_knn_probabilities_with_params(n, &knn, perp, 2.0, 50, 1e-5);
    let perplexities = vec![perp; n];
    let actual = compute_knn_probabilities_with_perplexities(n, &knn, &perplexities, 2.0, 50, 1e-5)
        .expect("valid adaptive perplexity input");
    assert_sparse_p_bit_equal(&actual, &expected, "uniform_perplexities_vs_scalar");

    // Reject out-of-range neighbor indices, non-finite distances, and negative distances
    let mut bad_oob = knn.clone();
    bad_oob[0].push((99, 0.2));
    assert!(
        compute_knn_probabilities_with_perplexities(n, &bad_oob, &perplexities, 2.0, 50, 1e-5)
            .is_err()
    );
    let mut bad_nan = knn.clone();
    bad_nan[0].push((2, f64::NAN));
    assert!(
        compute_knn_probabilities_with_perplexities(n, &bad_nan, &perplexities, 2.0, 50, 1e-5)
            .is_err()
    );
    let mut bad_neg = knn;
    bad_neg[0].push((2, -0.5));
    assert!(
        compute_knn_probabilities_with_perplexities(n, &bad_neg, &perplexities, 2.0, 50, 1e-5)
            .is_err()
    );
}

#[test]
fn test_compute_knn_probabilities_with_perplexities_variable_rows_and_clamping() {
    // Two nodes with identical non-degenerate distance rows should reflect different target perplexities.
    let base_row = vec![(2, 0.1), (3, 0.5), (4, 1.0), (5, 2.0), (6, 4.0), (7, 8.0)];
    let mut row1 = base_row.clone();
    row1[0].0 = 2; // neighbors 2..7 for both 0 and 1
    let knn: Vec<Vec<(usize, f64)>> = vec![
        base_row,                 // row 0: m=6, low perplexity (clamped from 0.5 -> 1.0)
        row1,                     // row 1: m=6, higher perplexity 4.5
        vec![(0, 0.1), (1, 0.2)], // row 2: m=2, perplexity 100.0 clamped to 1.99
        vec![(0, 0.5)],           // row 3: m=1
        vec![(0, 1.0), (1, 1.0)], // row 4: m=2, equal distances -> uniform
        vec![],                   // row 5: m=0 (still receives symmetric entries)
        vec![(0, 4.0), (1, 4.0)], // row 6: m=2
        vec![(0, 8.0), (1, 8.0)], // row 7: m=2
    ];
    let n = knn.len();
    let perplexities = vec![0.5, 4.5, 100.0, 1.5, 3.0, 2.0, 1.2, 2.5];
    let sparse_p =
        compute_knn_probabilities_with_perplexities(n, &knn, &perplexities, 2.0, 50, 1e-5)
            .expect("valid heterogeneous perplexities");

    assert_eq!(sparse_p.len(), n);
    let mut total_sum = 0.0;
    for (i, row) in sparse_p.iter().enumerate() {
        let mut prev = None;
        for &(j, p_ij) in row {
            assert_ne!(i, j);
            assert!(p_ij.is_finite() && p_ij > 0.0);
            if let Some(p) = prev {
                assert!(p < j);
            }
            prev = Some(j);
            let p_ji = sparse_p[j]
                .iter()
                .find(|&&(col, _)| col == i)
                .map(|&(_, v)| v)
                .expect("missing symmetric entry");
            assert!((p_ij - p_ji).abs() < 1e-12);
            total_sum += p_ij;
        }
    }
    // 7 of 8 rows are non-empty, so total probability sum is 7 / 8.
    assert!((total_sum - 7.0 / 8.0).abs() < 1e-6);

    // Node 0 had lower perplexity (0.5 -> 1.0) than node 1 (4.5) on identical neighbor distances 2..7,
    // so node 0 places strictly higher probability mass on closest neighbor 2 than node 1 does.
    let p02 = sparse_p[0].iter().find(|&&(j, _)| j == 2).unwrap().1;
    let p12 = sparse_p[1].iter().find(|&&(j, _)| j == 2).unwrap().1;
    assert!(p02 > p12, "expected p02 ({p02}) > p12 ({p12})");
}

#[test]
fn test_knn_from_csr_and_csr_probabilities_equivalence_and_validation() {
    let indptr = vec![0, 3, 4, 4, 7];
    let indices = vec![1, 2, 1, 0, 3, 2, 1];
    let distances = vec![0.4, 1.2, 0.4, 0.4, 0.0, 0.8, 1.5];
    let perplexities = vec![1.5, 2.0, 1.0, 2.5];

    let knn = knn_from_csr(4, &indptr, &indices, &distances).expect("valid CSR");
    let via_knn =
        compute_knn_probabilities_with_perplexities(4, &knn, &perplexities, 2.0, 50, 1e-5)
            .expect("valid knn");
    let via_csr = compute_knn_probabilities_from_csr(
        4,
        &indptr,
        &indices,
        &distances,
        &perplexities,
        2.0,
        50,
        1e-5,
    )
    .expect("valid csr");
    assert_sparse_p_bit_equal(&via_csr, &via_knn, "csr_vs_knn");

    // Edge cases: n = 0 and n = 1
    assert!(
        compute_knn_probabilities_from_csr::<f64>(0, &[0], &[], &[], &[], 2.0, 50, 1e-5)
            .expect("n=0 valid")
            .is_empty()
    );
    assert_eq!(
        compute_knn_probabilities_from_csr::<f64>(1, &[0, 0], &[], &[], &[1.0], 2.0, 50, 1e-5)
            .expect("n=1 valid"),
        vec![Vec::<(usize, f64)>::new()]
    );

    // Validation errors (must fail cleanly even for n < 2)
    assert!(knn_from_csr::<f64>(4, &indptr[..4], &indices, &distances).is_err());
    assert!(knn_from_csr::<f64>(4, &[1, 3, 4, 4, 7], &indices, &distances).is_err());
    assert!(knn_from_csr::<f64>(4, &[0, 4, 3, 4, 7], &indices, &distances).is_err());
    assert!(knn_from_csr::<f64>(4, &indptr, &indices[..6], &distances).is_err());
    assert!(knn_from_csr::<f64>(4, &indptr, &indices, &distances[..6]).is_err());
    assert!(knn_from_csr::<f64>(4, &[0, 10_000, 10_000, 10_000, 10_000], &[], &[]).is_err());

    let mut bad_idx = indices.clone();
    bad_idx[0] = 4;
    assert!(knn_from_csr::<f64>(4, &indptr, &bad_idx, &distances).is_err());

    for bad_d in [-0.1, f64::NAN, f64::INFINITY] {
        let mut bad_dists = distances.clone();
        bad_dists[0] = bad_d;
        assert!(knn_from_csr::<f64>(4, &indptr, &indices, &bad_dists).is_err());
    }

    // Perplexity validation errors (including n = 1)
    assert!(compute_knn_probabilities_from_csr(
        4,
        &indptr,
        &indices,
        &distances,
        &perplexities[..3],
        2.0,
        50,
        1e-5
    )
    .is_err());
    for bad_p in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(compute_knn_probabilities_from_csr::<f64>(
            1,
            &[0, 0],
            &[],
            &[],
            &[bad_p],
            2.0,
            50,
            1e-5
        )
        .is_err());
    }
    assert!(compute_knn_probabilities_with_perplexities::<f64>(
        2,
        &[vec![]],
        &[1.0, 1.0],
        2.0,
        50,
        1e-5
    )
    .is_err());
}
