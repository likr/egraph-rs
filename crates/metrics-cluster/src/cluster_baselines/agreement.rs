use egraph_core::{CoreError, CoreResult};

use super::{PartitionAgreement, compact_labels};

#[inline]
fn sign_preserving_clamp(x: f64) -> f64 {
    if x < 0.0 {
        x.min(-f64::EPSILON)
    } else {
        x.max(f64::EPSILON)
    }
}

fn group_marginal_multiplicities(counts: &[usize]) -> Vec<(usize, usize)> {
    let mut sorted = counts.to_vec();
    sorted.sort_unstable();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < sorted.len() {
        let val = sorted[i];
        let mut j = i + 1;
        while j < sorted.len() && sorted[j] == val {
            j += 1;
        }
        if val > 0 {
            out.push((val, j - i));
        }
        i = j;
    }
    out
}

/// Computes exact partition agreement metrics (`ARI`, `AMI`, `Homogeneity`, `Completeness`,
/// `V-Measure`, `FMI`) matching `scikit-learn` conventions and edge cases.
pub fn partition_agreement(
    labels_true: &[u32],
    labels_pred: &[u32],
) -> CoreResult<PartitionAgreement> {
    if labels_true.len() != labels_pred.len() {
        return Err(CoreError::dimension_mismatch(
            labels_true.len(),
            labels_pred.len(),
            "partition_agreement",
        ));
    }

    let n = labels_true.len();
    if n == 0 {
        return Ok(PartitionAgreement {
            ari: 1.0,
            ami: 1.0,
            homogeneity: 1.0,
            completeness: 1.0,
            v_measure: 1.0,
            fmi: 0.0,
        });
    }

    let (u_comp, a_counts, r_classes) = compact_labels(labels_true);
    let (v_comp, b_counts, c_classes) = compact_labels(labels_pred);

    let mut packed = Vec::with_capacity(n);
    for i in 0..n {
        packed.push(((u_comp[i] as u64) << 32) | (v_comp[i] as u64));
    }
    packed.sort_unstable();

    let mut contingency: Vec<(usize, usize, usize)> = Vec::new();
    let mut idx = 0usize;
    while idx < n {
        let key = packed[idx];
        let mut end = idx + 1;
        while end < n && packed[end] == key {
            end += 1;
        }
        let r = (key >> 32) as usize;
        let c = (key & 0xFFFF_FFFF) as usize;
        contingency.push((r, c, end - idx));
        idx = end;
    }

    let n_u128 = n as u128;
    let sum_nij_sq: u128 = contingency
        .iter()
        .map(|&(_, _, c)| (c as u128) * (c as u128))
        .sum();
    let sum_a_sq: u128 = a_counts.iter().map(|&c| (c as u128) * (c as u128)).sum();
    let sum_b_sq: u128 = b_counts.iter().map(|&c| (c as u128) * (c as u128)).sum();

    let tp = sum_nij_sq - n_u128;
    let fp = sum_b_sq - n_u128 - tp;
    let fn_ = sum_a_sq - n_u128 - tp;
    let tn = n_u128 * (n_u128 - 1) - tp - fp - fn_;

    let ari = if fn_ == 0 && fp == 0 {
        1.0
    } else {
        let num = match (tp.checked_mul(tn), fn_.checked_mul(fp)) {
            (Some(lhs), Some(rhs)) => {
                if lhs >= rhs {
                    2.0 * ((lhs - rhs) as f64)
                } else {
                    -2.0 * ((rhs - lhs) as f64)
                }
            }
            _ => 2.0 * ((tp as f64) * (tn as f64) - (fn_ as f64) * (fp as f64)),
        };
        let den =
            ((tp + fn_) as f64) * ((fn_ + tn) as f64) + ((tp + fp) as f64) * ((fp + tn) as f64);
        if den == 0.0 { 1.0 } else { num / den }
    };

    let fmi = if tp == 0 {
        0.0
    } else {
        let pk = (sum_b_sq - n_u128) as f64;
        let qk = (sum_a_sq - n_u128) as f64;
        let tk = tp as f64;
        (tk / pk).sqrt() * (tk / qk).sqrt()
    };

    let n_f = n as f64;
    let mut h_true = 0.0_f64;
    for &a_i in &a_counts {
        if a_i > 0 {
            let p = a_i as f64 / n_f;
            h_true -= p * p.ln();
        }
    }
    let mut h_pred = 0.0_f64;
    for &b_j in &b_counts {
        if b_j > 0 {
            let p = b_j as f64 / n_f;
            h_pred -= p * p.ln();
        }
    }

    let mut mi = 0.0_f64;
    let ln_n = n_f.ln();
    for &(r, c, nij) in &contingency {
        let p = nij as f64 / n_f;
        let log_term =
            (nij as f64).ln() + ln_n - (a_counts[r] as f64).ln() - (b_counts[c] as f64).ln();
        mi += p * log_term;
    }
    let mi = mi.max(0.0);

    let homogeneity = if h_true > 0.0 { mi / h_true } else { 1.0 };
    let completeness = if h_pred > 0.0 { mi / h_pred } else { 1.0 };
    let v_measure = if homogeneity + completeness == 0.0 {
        0.0
    } else {
        2.0 * homogeneity * completeness / (homogeneity + completeness)
    };

    let ami = if r_classes == 1 && c_classes == 1 {
        1.0
    } else if r_classes == 1 || c_classes == 1 {
        0.0
    } else {
        let mut ln_fact = vec![0.0_f64; n + 1];
        for k in 1..=n {
            ln_fact[k] = ln_fact[k - 1] + (k as f64).ln();
        }
        let a_groups = group_marginal_multiplicities(&a_counts);
        let b_groups = group_marginal_multiplicities(&b_counts);

        let mut emi = 0.0_f64;
        for &(a_i, mult_a) in &a_groups {
            for &(b_j, mult_b) in &b_groups {
                let mult = (mult_a as f64) * (mult_b as f64);
                let n_min = 1usize.max((a_i + b_j).saturating_sub(n));
                let n_max = a_i.min(b_j);
                let base_log =
                    ln_fact[a_i] + ln_fact[b_j] + ln_fact[n - a_i] + ln_fact[n - b_j] - ln_fact[n];
                let ln_ab = (a_i as f64 * b_j as f64).ln();
                for nij in n_min..=n_max {
                    let log_p = base_log
                        - ln_fact[nij]
                        - ln_fact[a_i - nij]
                        - ln_fact[b_j - nij]
                        - ln_fact[(n - a_i) + nij - b_j];
                    let term_log = (n_f * nij as f64).ln() - ln_ab;
                    emi += mult * (nij as f64 / n_f) * term_log * log_p.exp();
                }
            }
        }
        let normalizer = 0.5 * (h_true + h_pred);
        let num = sign_preserving_clamp(mi - emi);
        let den = sign_preserving_clamp(normalizer - emi);
        num / den
    };

    Ok(PartitionAgreement {
        ari,
        ami,
        homogeneity,
        completeness,
        v_measure,
        fmi,
    })
}
