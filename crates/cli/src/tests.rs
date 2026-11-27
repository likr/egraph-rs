use std::path::PathBuf;

use clap::Parser;
use egraph::prelude::MetricTarget;

use crate::io::CliError;
use crate::{
    Cli, ClusterMethod, ClusterRefineArg, Commands, InitMethod, LayoutArgs, LayoutMethod,
    ManifoldArg, OverlapRemovalArg, execute_layout, parse_metric_targets,
};

#[test]
fn parse_metric_targets_expands_all_with_dedup_and_validation() {
    let all_only = parse_metric_targets("all").expect("all");
    assert_eq!(all_only.len(), 22);
    assert!(!all_only.contains(&MetricTarget::ClusterAreaFaithfulness));

    let all_gcaf = parse_metric_targets("all,gcaf").expect("all,gcaf");
    assert_eq!(all_gcaf.len(), 23);
    assert_eq!(
        all_gcaf.last(),
        Some(&MetricTarget::ClusterAreaFaithfulness)
    );

    let gcaf_all = parse_metric_targets("gcaf,all").expect("gcaf,all");
    assert_eq!(gcaf_all.len(), 23);
    assert_eq!(
        gcaf_all.first(),
        Some(&MetricTarget::ClusterAreaFaithfulness)
    );

    let sns_all = parse_metric_targets("sns,all,all,gcaf,gcaf").expect("dedup");
    assert_eq!(sns_all.len(), 23);
    assert_eq!(sns_all.first(), Some(&MetricTarget::ScaleNormalizedStress));
    assert_eq!(sns_all.last(), Some(&MetricTarget::ClusterAreaFaithfulness));

    let err = parse_metric_targets("all,invalid_target").expect_err("trailing invalid target");
    assert!(matches!(err, CliError::InvalidInput(_)));

    let patch_nig = parse_metric_targets("adjusted-patch-nig,patch_nig").expect("patch-nig");
    assert_eq!(patch_nig, vec![MetricTarget::AdjustedPatchNig]);

    let extra_targets = parse_metric_targets(
        "geodesic-distortion,crossing-edges,torus-crossing-number,crossing-edges-torus,prism-distortion,nor-metrics,nor",
    )
    .expect("new metric targets");
    assert_eq!(
        extra_targets,
        vec![
            MetricTarget::GeodesicDistortion,
            MetricTarget::CrossingEdges,
            MetricTarget::TorusCrossingNumber,
            MetricTarget::CrossingEdgesTorus,
            MetricTarget::PrismDistortion,
            MetricTarget::NorMetrics,
        ]
    );
}

#[test]
fn parse_metric_targets_parses_all_18_eval_baseline_targets_and_aliases() {
    let all_only = parse_metric_targets("all").expect("all");
    assert_eq!(all_only.len(), 22);

    let all_18 = parse_metric_targets(
        "dsc,dc,cdm,hdm,sepme-knng,sepme-rng,sepme-gab,sepme-gong,neighborhood-hit,silhouette,calinski-harabasz,davies-bouldin,cq-kmeans-ari,cq-kmeans-ami,cq-hac-hom,cq-hac-ami,pairwise-distance-overlap-error,sprawlter-nn",
    )
    .expect("all 18 eval-baselines targets");
    assert_eq!(
        all_18,
        vec![
            MetricTarget::DistanceConsistency,
            MetricTarget::DistributionConsistency,
            MetricTarget::ClassDensityMeasure,
            MetricTarget::HistogramDensityMeasure,
            MetricTarget::SepmeKnng,
            MetricTarget::SepmeRng,
            MetricTarget::SepmeGab,
            MetricTarget::SepmeGong,
            MetricTarget::NeighborhoodHit,
            MetricTarget::SilhouetteScore,
            MetricTarget::CalinskiHarabaszScore,
            MetricTarget::DaviesBouldinScore,
            MetricTarget::CqKmeansAri,
            MetricTarget::CqKmeansAmi,
            MetricTarget::CqHacHom,
            MetricTarget::CqHacAmi,
            MetricTarget::PairwiseDistanceOverlapError,
            MetricTarget::SprawlterNn,
        ]
    );

    let sepme_aliases = parse_metric_targets("sepme-knng,sepme_knng,sepme-knng2,sepme_knng2")
        .expect("sepme aliases");
    assert_eq!(sepme_aliases, vec![MetricTarget::SepmeKnng]);

    let snake_and_short = parse_metric_targets(
        "distance_consistency,distribution_consistency,class_density_measure,2d_hdm,histogram_density_measure,sepme_rng,sepme_gab,sepme_gong,neighborhood_hit,m_nh,nh,silhouette_score,calinski_harabasz,calinski_harabasz_score,ch,davies_bouldin,davies_bouldin_score,db,cq_kmeans_ari,cq_kmeans_ami,cq_hac_hom,cq_hac_ami,pairwise_distance_overlap_error,pdoe,sprawlter_nn,sprawlter",
    )
    .expect("snake_case and alias parity");
    assert_eq!(
        snake_and_short,
        vec![
            MetricTarget::DistanceConsistency,
            MetricTarget::DistributionConsistency,
            MetricTarget::ClassDensityMeasure,
            MetricTarget::HistogramDensityMeasure,
            MetricTarget::SepmeRng,
            MetricTarget::SepmeGab,
            MetricTarget::SepmeGong,
            MetricTarget::NeighborhoodHit,
            MetricTarget::SilhouetteScore,
            MetricTarget::CalinskiHarabaszScore,
            MetricTarget::DaviesBouldinScore,
            MetricTarget::CqKmeansAri,
            MetricTarget::CqKmeansAmi,
            MetricTarget::CqHacHom,
            MetricTarget::CqHacAmi,
            MetricTarget::PairwiseDistanceOverlapError,
            MetricTarget::SprawlterNn,
        ]
    );
}

#[test]
fn cli_parses_cluster_and_overlap_removal_flags() {
    let cli_prism = Cli::try_parse_from(["egraph", "layout", "--dataset", "karate", "--prism"])
        .expect("--prism");
    match cli_prism.command {
        Commands::Layout(args) => {
            assert!(args.prism);
            assert_eq!(args.overlap_removal, None);
            assert_eq!(args.cluster_refine, ClusterRefineArg::None);
        }
        _ => panic!("expected Layout command"),
    }

    for (flag, expected) in [
        ("none", OverlapRemovalArg::None),
        ("prism", OverlapRemovalArg::Prism),
        ("gtree", OverlapRemovalArg::Gtree),
        ("forbid", OverlapRemovalArg::Forbid),
        ("forbid-prime", OverlapRemovalArg::ForbidPrime),
        ("sparse-forbid", OverlapRemovalArg::SparseForbid),
        ("twin-declutter", OverlapRemovalArg::TwinDeclutter),
    ] {
        let parsed = Cli::try_parse_from([
            "egraph",
            "layout",
            "--dataset",
            "karate",
            "--overlap-removal",
            flag,
            "--overlap-radius",
            "0.1",
        ])
        .expect("parse overlap-removal");
        match parsed.command {
            Commands::Layout(args) => {
                assert!(!args.prism);
                assert_eq!(args.overlap_removal, Some(expected));
                assert!((args.prism_radius - 0.1).abs() < 1e-12);
            }
            _ => panic!("expected Layout command"),
        }
    }

    let conflict = Cli::try_parse_from([
        "egraph",
        "layout",
        "--dataset",
        "karate",
        "--prism",
        "--overlap-removal",
        "gtree",
    ]);
    assert!(
        conflict.is_err(),
        "expected clap conflict between --prism and --overlap-removal"
    );

    for (method_str, expected) in [
        ("linlog", LayoutMethod::Linlog),
        ("fr", LayoutMethod::Fr),
        ("fa2-overlap", LayoutMethod::Fa2Overlap),
        ("backbone", LayoutMethod::Backbone),
        ("gib", LayoutMethod::Gib),
    ] {
        let parsed = Cli::try_parse_from([
            "egraph",
            "layout",
            "--dataset",
            "karate",
            "--method",
            method_str,
            "--cluster-refine",
            "hac",
            "--labels",
            "labels.json",
        ])
        .expect("parse layout method and cluster-refine");
        match parsed.command {
            Commands::Layout(args) => {
                assert_eq!(args.method, expected);
                assert_eq!(args.cluster_refine, ClusterRefineArg::Hac);
                assert_eq!(args.labels, Some(PathBuf::from("labels.json")));
            }
            _ => panic!("expected Layout command"),
        }
    }

    for (method_str, expected) in [
        ("leiden-mod", ClusterMethod::LeidenMod),
        ("leiden-cpm", ClusterMethod::LeidenCpm),
    ] {
        let parsed = Cli::try_parse_from([
            "egraph",
            "cluster",
            "--dataset",
            "karate",
            "--method",
            method_str,
            "--resolution",
            "1.25",
            "--randomness",
            "0.05",
        ])
        .expect("parse leiden cluster flags");
        match parsed.command {
            Commands::Cluster(args) => {
                assert_eq!(args.method, expected);
                assert!((args.resolution - 1.25).abs() < 1e-12);
                assert!((args.randomness - 0.05).abs() < 1e-12);
            }
            _ => panic!("expected Cluster command"),
        }
    }
}

#[test]
fn cli_parses_extended_layout_methods_and_pfs_prime_flags() {
    for (method_str, expected) in [
        ("hde", LayoutMethod::Hde),
        ("sgd-ortmann", LayoutMethod::SgdOrtmann),
        ("maxent", LayoutMethod::Maxent),
        ("largevis", LayoutMethod::Largevis),
        ("drgraph", LayoutMethod::Drgraph),
        ("graph-cne", LayoutMethod::GraphCne),
        ("grip", LayoutMethod::Grip),
        ("gem", LayoutMethod::Gem),
        ("sel", LayoutMethod::Sel),
        ("sfc", LayoutMethod::Sfc),
    ] {
        let parsed_layout = Cli::try_parse_from([
            "egraph",
            "layout",
            "--dataset",
            "karate",
            "--method",
            method_str,
            "--overlap-removal",
            "pfs-prime",
            "--overlap-radius",
            "0.12",
            "--seed",
            "99",
        ])
        .expect("parse extended layout method and pfs-prime");
        match parsed_layout.command {
            Commands::Layout(args) => {
                assert_eq!(args.method, expected);
                assert_eq!(args.overlap_removal, Some(OverlapRemovalArg::PfsPrime));
                assert!((args.prism_radius - 0.12).abs() < 1e-12);
                assert_eq!(args.seed, 99);
            }
            _ => panic!("expected Layout command"),
        }

        let parsed_bench = Cli::try_parse_from([
            "egraph",
            "bench",
            "--dataset",
            "karate",
            "--method",
            method_str,
            "--seed",
            "99",
        ])
        .expect("parse extended bench method");
        match parsed_bench.command {
            Commands::Bench(args) => {
                assert_eq!(args.method, expected);
                assert_eq!(args.seed, 99);
            }
            _ => panic!("expected Bench command"),
        }
    }

    let conflict = Cli::try_parse_from([
        "egraph",
        "layout",
        "--dataset",
        "karate",
        "--prism",
        "--overlap-removal",
        "pfs-prime",
    ]);
    assert!(
        conflict.is_err(),
        "expected clap conflict between --prism and --overlap-removal pfs-prime"
    );
}

#[test]
fn cli_parses_manifold_nd_and_schedule_window_flags() {
    let parsed_tsnet_rn = Cli::try_parse_from([
        "egraph",
        "layout",
        "--dataset",
        "karate",
        "--method",
        "tsnet-rn",
        "--dims",
        "3",
    ])
    .expect("parse tsnet-rn and dims");
    match parsed_tsnet_rn.command {
        Commands::Layout(args) => {
            assert_eq!(args.method, LayoutMethod::TsnetRn);
            assert_eq!(args.dims, 3);
            assert_eq!(args.manifold, ManifoldArg::None);
        }
        _ => panic!("expected Layout command"),
    }

    for (manifold_str, expected) in [
        ("none", ManifoldArg::None),
        ("hyperbolic", ManifoldArg::Hyperbolic),
        ("hyperbolic-lorentz", ManifoldArg::HyperbolicLorentz),
        ("spherical", ManifoldArg::Spherical),
        ("torus", ManifoldArg::Torus),
    ] {
        let parsed = Cli::try_parse_from([
            "egraph",
            "layout",
            "--dataset",
            "karate",
            "--method",
            "sgd-full",
            "--manifold",
            manifold_str,
            "--manifold-scale",
            "1.5",
            "--torus-aspect",
            "2.0",
            "--torus-auto-pan",
            "--schedule-windows",
            "2",
            "--schedule-window-iters",
            "10",
            "--manifold-rescale-final",
            "2.5",
        ])
        .expect("parse manifold and schedule-window flags");
        match parsed.command {
            Commands::Layout(args) => {
                assert_eq!(args.manifold, expected);
                assert!((args.manifold_scale - 1.5).abs() < 1e-12);
                assert!((args.torus_aspect - 2.0).abs() < 1e-12);
                assert!(args.torus_auto_pan);
                assert_eq!(args.schedule_windows, Some(2));
                assert_eq!(args.schedule_window_iters, Some(10));
                assert_eq!(args.manifold_rescale_final, Some(2.5));
            }
            _ => panic!("expected Layout command"),
        }
    }
}

#[test]
fn execute_layout_rejects_conflicting_prism_and_overlap_removal_for_rust_callers() {
    for mode in [OverlapRemovalArg::Gtree, OverlapRemovalArg::PfsPrime] {
        let args = LayoutArgs {
            input: None,
            dataset: Some("karate".to_string()),
            method: LayoutMethod::Stress,
            init: InitMethod::PivotMds,
            dims: 2,
            manifold: ManifoldArg::None,
            manifold_scale: 1.0,
            torus_aspect: 1.0,
            torus_auto_pan: false,
            schedule_windows: None,
            schedule_window_iters: None,
            manifold_rescale_final: None,
            labels: None,
            refine: false,
            cluster_refine: ClusterRefineArg::None,
            prism: true,
            overlap_removal: Some(mode),
            prism_radius: 0.05,
            seed: 42,
            output: None,
        };
        let err = execute_layout(&args).expect_err("runtime conflict guard");
        assert!(matches!(err, CliError::InvalidInput(_)));
    }
}
