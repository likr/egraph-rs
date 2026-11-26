use super::*;
use crate::csr::CsrGraph;
use crate::error::CoreError;

fn torus_grid_csr(rows: usize, cols: usize) -> CsrGraph<f64> {
    let n = rows * cols;
    let mut edges = Vec::with_capacity(2 * n);
    for r in 0..rows {
        for c in 0..cols {
            let u = (r * cols + c) as u32;
            let right = (r * cols + ((c + 1) % cols)) as u32;
            let down = (((r + 1) % rows) * cols + c) as u32;
            edges.push((u, right));
            edges.push((u, down));
        }
    }
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

fn count_wrapped_edges_1d(coords: &[f64], edges: &[(u32, u32)]) -> usize {
    edges
        .iter()
        .filter(|&&(u, v)| (coords[u as usize] - coords[v as usize]).abs() > 0.5)
        .count()
}

fn brute_force_min_wrapped_1d(coords: &[f64], edges: &[(u32, u32)]) -> usize {
    let mut sorted = coords.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    sorted.dedup();
    if sorted.len() <= 1 {
        return 0;
    }
    let mut min_wrap = usize::MAX;
    for k in 0..sorted.len() {
        let next = if k + 1 < sorted.len() {
            sorted[k + 1]
        } else {
            1.0 + sorted[0]
        };
        let cut = (0.5 * (sorted[k] + next)).rem_euclid(1.0);
        let shifted: Vec<f64> = coords.iter().map(|&x| (x - cut).rem_euclid(1.0)).collect();
        min_wrap = min_wrap.min(count_wrapped_edges_1d(&shifted, edges));
    }
    min_wrap
}

#[test]
fn test_torus_spectral_phase_init_small_and_large() {
    // Empty and singleton boundary cases.
    let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let m0 = ManifoldLayout2d::<f64>::torus_spectral_phase_init(&g0, 42).unwrap();
    assert_eq!(m0.len(), 0);
    assert_eq!(m0.kind(), ManifoldKind::Torus2d);

    let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let m1 = ManifoldLayout2d::<f64>::torus_spectral_phase_init(&g1, 42).unwrap();
    assert_eq!(m1.len(), 1);

    // Small torus grid (4x4 = 16 <= 48, dense Jacobi path).
    let g16 = torus_grid_csr(4, 4);
    let m16_a = ManifoldLayout2d::<f64>::torus_spectral_phase_init(&g16, 42).unwrap();
    let m16_b = ManifoldLayout2d::<f64>::torus_spectral_phase_init(&g16, 42).unwrap();
    assert_eq!(m16_a, m16_b);
    assert_eq!(m16_a.kind(), ManifoldKind::Torus2d);
    for (&x, &y) in m16_a.x().iter().zip(m16_a.y().iter()) {
        assert!((0.0..1.0).contains(&x) && (0.0..1.0).contains(&y));
    }

    // Larger torus grid (8x8 = 64 > 48, SpMV block thick-restart Lanczos path) in f32 and f64.
    let g64 = torus_grid_csr(8, 8);
    let m64 = ManifoldLayout2d::<f32>::torus_spectral_phase_init(&g64, 7).unwrap();
    assert_eq!(m64.len(), 64);
    for (&x, &y) in m64.x().iter().zip(m64.y().iter()) {
        assert!((0.0_f32..1.0).contains(&x) && (0.0_f32..1.0).contains(&y));
    }
}

#[test]
fn test_torus_auto_pan_optimal_cut_and_isometry() {
    // Cluster split across the periodic boundary x=0 and y=0 can be unwrapped to 0 wrapped edges.
    let g_split =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], None, false, false)
            .unwrap();
    let mut ml_split = ManifoldLayout2d::<f64>::from_xy(
        ManifoldKind::Torus2d,
        vec![0.88, 0.94, 0.06, 0.12],
        vec![0.90, 0.05, 0.08, 0.92],
    )
    .unwrap();
    ml_split.set_torus_extents(2.5, 0.5).unwrap();
    let edges_split = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
    assert!(count_wrapped_edges_1d(ml_split.x(), &edges_split) > 0);
    assert!(count_wrapped_edges_1d(ml_split.y(), &edges_split) > 0);

    let panned = ml_split.torus_auto_pan(&g_split).unwrap();
    assert_eq!(panned.metric(), ml_split.metric());
    assert_eq!(count_wrapped_edges_1d(panned.x(), &edges_split), 0);
    assert_eq!(count_wrapped_edges_1d(panned.y(), &edges_split), 0);
    for i in 0..4 {
        for j in (i + 1)..4 {
            let da = ml_split.geodesic_distance(i, j).unwrap();
            let db = panned.geodesic_distance(i, j).unwrap();
            assert!((da - db).abs() < 1e-12);
        }
    }

    // Randomized brute-force oracle check on torus graphs.
    for seed in 1..=12u64 {
        let g = torus_grid_csr(4, 4);
        let mut ml = ManifoldLayout2d::<f64>::init(ManifoldKind::Torus2d, 16, seed, 1.0).unwrap();
        ml.set_torus_extents(1.75, 0.8).unwrap();
        let p = ml.torus_auto_pan(&g).unwrap();
        let mut edges = Vec::new();
        for u in 0..g.num_nodes() {
            for &v in g.neighbors(u) {
                if (u as u32) < v {
                    edges.push((u as u32, v));
                }
            }
        }
        assert_eq!(
            count_wrapped_edges_1d(p.x(), &edges),
            brute_force_min_wrapped_1d(ml.x(), &edges)
        );
        assert_eq!(
            count_wrapped_edges_1d(p.y(), &edges),
            brute_force_min_wrapped_1d(ml.y(), &edges)
        );
        for i in 0..16 {
            for j in (i + 1)..16 {
                let da = ml.geodesic_distance(i, j).unwrap();
                let db = p.geodesic_distance(i, j).unwrap();
                assert!((da - db).abs() < 1e-10);
            }
        }
    }

    // Rejects non-torus and mismatched node counts.
    let sph = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Spherical, 4);
    assert!(matches!(
        sph.torus_auto_pan(&g_split),
        Err(CoreError::UnsupportedOperation(_))
    ));
    let wrong_len = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Torus2d, 3);
    assert!(matches!(
        wrong_len.torus_auto_pan(&g_split),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // Directed graph with reciprocal edges + isolated node works cleanly.
    let g_dir = CsrGraph::<f64>::from_edges(
        5,
        &[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 0)],
        None,
        true,
        false,
    )
    .unwrap();
    let m_dir = ManifoldLayout2d::<f64>::torus_spectral_phase_init(&g_dir, 13).unwrap();
    assert_eq!(m_dir.len(), 5);
    let p_dir = m_dir.torus_auto_pan(&g_dir).unwrap();
    assert_eq!(p_dir.len(), 5);
}
