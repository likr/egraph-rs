use super::matrix::CsrMatrix;

/// Computes the strength of connection matrix $S$ from $A$.
/// Classical strength: nodes $i$ and $j$ are strongly connected if
/// $|A_{ij}| \ge \theta \max_{k \neq i} |A_{ik}|$
pub fn strength_of_connection<S>(a: &CsrMatrix<S>, theta: S) -> CsrMatrix<S>
where
    S: Copy + num_traits::Float + num_traits::Zero + std::ops::AddAssign + Default + std::cmp::PartialOrd,
{
    let mut indptr = vec![0; a.rows + 1];
    let mut indices = Vec::new();
    let mut data = Vec::new();

    for i in 0..a.rows {
        // Find max off-diagonal
        let mut max_off_diag = S::zero();
        for j_ptr in a.indptr[i]..a.indptr[i + 1] {
            let j = a.indices[j_ptr];
            if i != j {
                let val = a.data[j_ptr].abs();
                if val > max_off_diag {
                    max_off_diag = val;
                }
            }
        }

        let threshold = theta * max_off_diag;
        let mut row_nnz = 0;

        for j_ptr in a.indptr[i]..a.indptr[i + 1] {
            let j = a.indices[j_ptr];
            let val = a.data[j_ptr];

            if i == j || val.abs() >= threshold {
                indices.push(j);
                data.push(val);
                row_nnz += 1;
            }
        }
        indptr[i + 1] = indptr[i] + row_nnz;
    }

    CsrMatrix {
        rows: a.rows,
        cols: a.cols,
        indptr,
        indices,
        data,
    }
}

/// Computes aggregates using standard greedy aggregation on the strength matrix.
/// Returns a vector where `aggregates[i]` is the aggregate ID of node `i`.
pub fn standard_aggregation<S>(strength: &CsrMatrix<S>) -> (Vec<usize>, usize)
where
    S: Copy + num_traits::Float + num_traits::Zero + std::ops::AddAssign + Default,
{
    let n = strength.rows;
    let mut aggregates = vec![usize::MAX; n];
    let mut num_aggregates = 0;

    // First pass: create aggregates from unaggregated nodes
    for i in 0..n {
        if aggregates[i] != usize::MAX {
            continue;
        }

        // Check if node i has unaggregated neighbors
        let mut has_unagg_neighbors = false;
        for j_ptr in strength.indptr[i]..strength.indptr[i + 1] {
            let j = strength.indices[j_ptr];
            if i != j && aggregates[j] == usize::MAX {
                has_unagg_neighbors = true;
                break;
            }
        }

        if has_unagg_neighbors {
            aggregates[i] = num_aggregates;
            for j_ptr in strength.indptr[i]..strength.indptr[i + 1] {
                let j = strength.indices[j_ptr];
                if aggregates[j] == usize::MAX {
                    aggregates[j] = num_aggregates;
                }
            }
            num_aggregates += 1;
        }
    }

    // Second pass: put isolated nodes or unaggregated nodes into neighboring aggregates
    for i in 0..n {
        if aggregates[i] == usize::MAX {
            for j_ptr in strength.indptr[i]..strength.indptr[i + 1] {
                let j = strength.indices[j_ptr];
                if aggregates[j] != usize::MAX {
                    aggregates[i] = aggregates[j];
                    break;
                }
            }
            // If still unaggregated (completely isolated), put it in its own aggregate
            if aggregates[i] == usize::MAX {
                aggregates[i] = num_aggregates;
                num_aggregates += 1;
            }
        }
    }

    (aggregates, num_aggregates)
}

/// Creates a tentative prolongator $P_0$ (binary matrix) from aggregates.
pub fn tentative_prolongator<S>(aggregates: &[usize], num_aggregates: usize) -> CsrMatrix<S>
where
    S: Copy + num_traits::Float + num_traits::Zero + std::ops::AddAssign + Default,
{
    let mut counts = vec![S::zero(); num_aggregates];
    for &agg in aggregates {
        counts[agg] += S::one();
    }

    let n = aggregates.len();
    let mut indptr = vec![0; n + 1];
    let mut indices = vec![0; n];
    let mut data = vec![S::one(); n];

    for i in 0..n {
        indptr[i + 1] = i + 1;
        let agg = aggregates[i];
        indices[i] = agg;
        let c = counts[agg];
        if c > S::zero() {
            data[i] = S::one() / c.sqrt();
        } else {
            data[i] = S::zero();
        }
    }

    CsrMatrix {
        rows: n,
        cols: num_aggregates,
        indptr,
        indices,
        data,
    }
}
