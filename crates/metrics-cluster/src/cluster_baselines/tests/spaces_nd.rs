use super::super::separation::dir_cpt_from_degree_counts;
use super::super::space_dispatch::build_spatial_index;
use super::*;
use egraph_core::{DeterministicRng, LayoutNd};

fn brute_sepme_nd(layout: &LayoutNd<f64>, labels: &[u32], mode: u8, gamma: f64) -> f64 {
    let n = layout.num_nodes();
    let d = layout.dims();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return 1.0;
    }
    let coords = layout.coords();
    let (comp, sizes, k_classes) = compact_labels(labels);
    let mut deg = vec![0usize; n];
    let mut same = vec![0usize; n];
    let dist_sq = |a: &[f64], b: &[f64]| -> f64 { (0..d).map(|ax| (a[ax] - b[ax]).powi(2)).sum() };

    for i in 0..n {
        let pi = &coords[i * d..(i + 1) * d];
        for j in 0..n {
            if i == j || (mode != 2 && j < i) {
                continue;
            }
            let pj = &coords[j * d..(j + 1) * d];
            let d_ij_sq = dist_sq(pi, pj);
            if d_ij_sq <= 0.0 {
                continue;
            }
            let mut ok = true;
            if mode == 2 {
                let mut m = vec![0.0_f64; d];
                for ax in 0..d {
                    m[ax] = gamma * pj[ax] + (1.0 - gamma) * pi[ax];
                }
                let block_thresh = (1.0 - gamma).powi(2) * d_ij_sq * (1.0 + 1e-12);
                for q in 0..n {
                    if q != i && q != j && dist_sq(&coords[q * d..(q + 1) * d], &m) <= block_thresh
                    {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    deg[i] += 1;
                    if comp[i] == comp[j] {
                        same[i] += 1;
                    }
                }
            } else {
                for r in 0..n {
                    if r == i || r == j {
                        continue;
                    }
                    let pr = &coords[r * d..(r + 1) * d];
                    let d_ir_sq = dist_sq(pi, pr);
                    let d_jr_sq = dist_sq(pj, pr);
                    let blocked = if mode == 1 {
                        d_ir_sq + d_jr_sq <= d_ij_sq * (1.0 + 1e-12)
                    } else {
                        let t = d_ij_sq * (1.0 - 1e-12);
                        d_ir_sq < t && d_jr_sq < t
                    };
                    if blocked {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    deg[i] += 1;
                    deg[j] += 1;
                    if comp[i] == comp[j] {
                        same[i] += 1;
                        same[j] += 1;
                    }
                }
            }
        }
    }
    dir_cpt_from_degree_counts(&comp, &sizes, k_classes, &deg, &same)
}

#[test]
fn test_nd_across_dimensions_and_coincident_ties() {
    for dims in [1usize, 2, 3, 4, 10] {
        let mut rng = DeterministicRng::from_seed_u64(100 + dims as u64);
        let n = 12usize;
        let mut flat64 = Vec::with_capacity(n * dims);
        let mut labels = Vec::with_capacity(n);
        for c in 0..3u32 {
            let offset = (c as f64) * 20.0;
            for _ in 0..4 {
                for ax in 0..dims {
                    let jitter = (rng.next_f64() - 0.5) * 0.2;
                    let base = if ax == 0 { offset } else { 0.0 };
                    flat64.push(base + jitter);
                }
                labels.push(c * 5 + 3);
            }
        }
        let l64 = LayoutNd::from_flat(n, dims, &flat64).unwrap();
        let flat32: Vec<f32> = flat64.iter().map(|&v| v as f32).collect();
        let l32 = LayoutNd::from_flat(n, dims, &flat32).unwrap();

        assert_close(
            distance_consistency(&l64, &labels).unwrap(),
            1.0,
            1e-12,
            &format!("d{dims}.dsc"),
        );
        assert_close(
            distribution_consistency(&l64, &labels, 100).unwrap(),
            1.0,
            1e-12,
            &format!("d{dims}.dc_sparse_100"),
        );
        assert!(silhouette_score(&l64, &labels).unwrap() > 0.95);
        assert!(calinski_harabasz_score(&l64, &labels).unwrap() > 100.0);
        assert!(davies_bouldin_score(&l64, &labels).unwrap() < 0.1);
        assert_eq!(
            pairwise_distance_overlap_error(&l64, &labels, 20).unwrap(),
            0.0
        );
        assert_close(
            cq_kmeans(&l64, &labels, 5, 50, 42).unwrap().ari,
            1.0,
            1e-12,
            &format!("d{dims}.km"),
        );
        assert_close(
            cq_hac(&l64, &labels).unwrap().ari,
            1.0,
            1e-12,
            &format!("d{dims}.hac"),
        );
        assert_close(
            sepme_knng(&l64, &labels, 3).unwrap(),
            1.0,
            1e-12,
            &format!("d{dims}.knng"),
        );
        assert_close(
            neighborhood_hit(&l64, &labels, 3).unwrap(),
            1.0,
            1e-12,
            &format!("d{dims}.nh"),
        );
        assert_close(
            sepme_rng(&l64, &labels).unwrap(),
            brute_sepme_nd(&l64, &labels, 0, 0.0),
            1e-12,
            &format!("d{dims}.rng"),
        );
        assert_close(
            sepme_gab(&l64, &labels).unwrap(),
            brute_sepme_nd(&l64, &labels, 1, 0.0),
            1e-12,
            &format!("d{dims}.gab"),
        );
        assert_close(
            sepme_gong(&l64, &labels, 0.35).unwrap(),
            brute_sepme_nd(&l64, &labels, 2, 0.35),
            1e-12,
            &format!("d{dims}.gong"),
        );

        assert_close(
            distance_consistency(&l32, &labels).unwrap(),
            1.0,
            1e-12,
            &format!("d{dims}.f32.dsc"),
        );
        assert!(silhouette_score(&l32, &labels).unwrap() > 0.95);
        assert_close(
            cq_hac(&l32, &labels).unwrap().ari,
            1.0,
            1e-12,
            &format!("d{dims}.f32.hac"),
        );
        assert_close(
            sepme_knng(&l32, &labels, 3).unwrap(),
            1.0,
            1e-12,
            &format!("d{dims}.f32.knng"),
        );
    }

    for dims in [1usize, 3, 4, 10] {
        let mut rng = DeterministicRng::from_seed_u64(777 + dims as u64);
        let n = 36usize;
        let mut coords = Vec::with_capacity(n * dims);
        for i in 0..n {
            if i == 5 {
                for ax in 0..dims {
                    coords.push(coords[ax]);
                }
            } else {
                for _ in 0..dims {
                    coords.push((rng.next_f64() - 0.5) * 10.0);
                }
            }
        }
        let layout = LayoutNd::from_flat(n, dims, &coords).unwrap();
        let labels: Vec<u32> = (0..n).map(|i| (i % 4) as u32).collect();
        let idx = build_spatial_index(&layout).unwrap();

        assert_eq!(
            sepme_knng(&layout, &labels, 4).unwrap().to_bits(),
            sepme_knng_with_index(&layout, &labels, 4, idx.as_ref())
                .unwrap()
                .to_bits()
        );
        assert_eq!(
            neighborhood_hit(&layout, &labels, 7).unwrap().to_bits(),
            neighborhood_hit_with_index(&layout, &labels, 7, idx.as_ref())
                .unwrap()
                .to_bits()
        );
        assert_close(
            sepme_rng(&layout, &labels).unwrap(),
            brute_sepme_nd(&layout, &labels, 0, 0.0),
            1e-12,
            &format!("rand.d{dims}.rng"),
        );
        assert_close(
            sepme_gab(&layout, &labels).unwrap(),
            brute_sepme_nd(&layout, &labels, 1, 0.0),
            1e-12,
            &format!("rand.d{dims}.gab"),
        );
        for gamma in [0.0, 0.35, 0.5, 0.75] {
            assert_close(
                sepme_gong(&layout, &labels, gamma).unwrap(),
                brute_sepme_nd(&layout, &labels, 2, gamma),
                1e-12,
                &format!("rand.d{dims}.gong({gamma})"),
            );
        }
    }
}
