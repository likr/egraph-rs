# `egraph` v6 Python チュートリアル（Google Colab 対応）

`egraph` v6 は、**Pure Rust による高速・決定論的・シングルスレッド**なグラフレイアウト計算および品質評価ライブラリです。

> [!IMPORTANT]
> **役割の分離（レイアウト計算エンジンとしての `egraph`）**  
> `egraph` v6 は **グラフデータ構造（`CsrGraph`）・レイアウト計算（`Layout2d` / `LayoutNd` / `ManifoldLayout2d` / 各種ソルバ）・コミュニティ抽出／階層化・品質評価指標（`evaluate_metrics` -> `MetricReport` など）のみ** を提供します。  
> ネットワーク図の描画・可視化はすべて Python の標準的な可視化ライブラリ（`matplotlib`・`networkx`・`numpy`）を組み合わせて行います。

---

## Google Colab / ローカル環境でのセットアップ

- **動作要件**: Python `>= 3.10`、`numpy >= 1.26`
- **Google Colab** では各ノートブックの Colab バッジから直接起動し、先頭セルで次のインストールコマンドを実行してください（現在 v6 はプレリリース版のため `--pre` が必要です）。

### 標準インストール（NB02〜NB09）

```bash
pip install -q --pre "egraph==6.0.0a9" matplotlib-fontja
```

### 相互運用・可視化拡張付きインストール（NB01 推奨）

`NetworkX` / `SciPy` / `igraph` / `matplotlib` との相互運用をすべて試す場合は `[interop,viz]` extra を指定します。

```bash
pip install -q --pre "egraph[interop,viz]==6.0.0a9" matplotlib-fontja
```

- `matplotlib-fontja` は Google Colab 上の Matplotlib で日本語ラベルを文字化けなく表示するためのパッケージです（`import matplotlib_fontja` で自動適用されます）。ローカル環境等で未インストールの場合は `try: import matplotlib_fontja; except ImportError: pass` によりフォールバックできます。
- Google Colab 上で全セルをまとめて実行する場合は、「ランタイム」→「すべてのセルを実行」を選択してください。

---

## v5 から v6 への移行・関連ドキュメント

`egraph` v6 では従来の v5 レガシー API から、`CsrGraph` / `Layout2d` / 統一ソルバ API（`iterations` / `max_iterations` / `labels` / `MetricReport` / `Result` クラス）への破壊的変更が行われています。v5 からの移行手順と API 対応表は以下のドキュメントを参照してください。

- **v5 → v6 移行ガイド**: [`../docs/MIGRATION_V6.md`](../docs/MIGRATION_V6.md)
- **Python API ドキュメント（Sphinx）**: [https://egraph-py-docs.netlify.app/](https://egraph-py-docs.netlify.app/)

---

## ノートブック一覧（全 9 本ロードマップ）

トピックごとに以下の 9 本のチュートリアルノートブックで構成されています。

| # | ファイル名 | Google Colab | タイトル・扱う主なトピック | 状態 |
|---|---|---|---|---|
| 01 | [`01_getting_started_and_data_structures.ipynb`](01_getting_started_and_data_structures.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/likr/egraph-rs/blob/master/notebooks/01_getting_started_and_data_structures.ipynb) | **入門とデータ構造**: `CsrGraph`（`len`/`edges`/`has_edge`/`edge_weight`/`is_connected`/`weight_dtype`）・組み込みデータセット（`load_dataset`/`list_datasets`/`dataset_info`）＆合成グラフ生成 7 種・`parse_csr_csv`（2 列無重み／3 列重み付き両対応）・`Layout2d` / `LayoutNd`（`.dims`, `copy`, `get`/`set`, `normalize_std`）・`egraph.interop`（`to_networkx`/`to_igraph` 2D/N-D/Manifold 対応）・空間補助機能（`ArenaOctree3d`, `DynamicKdTree`, Delaunay・矩形重なり） | **利用可能 (Ready)** |
| 02 | [`02_initialization_and_stress_layouts.ipynb`](02_initialization_and_stress_layouts.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/likr/egraph-rs/blob/master/notebooks/02_initialization_and_stress_layouts.ipynb) | **初期配置とストレス系レイアウト**: `pivot_mds_init` / `pivot_mds_from_distances` / `koren_spectral_init` / `classical_mds` / `hde` / `hde_nd` / `rd_mds` / `representative_mds`・3D `LayoutNd` 最適化（`StressMajorizationSolver` / `MaxentStressSolver` 等）と可視化・`KamadaKawaiSolver` / `GripSolver`・`SgdSolver`（`iterations=` / `from_packed_symmetric` / キーワード専用 `from_pairs_with_hinges`）協調実行 | **利用可能 (Ready)** |
| 03 | [`03_neighborhood_preserving_layouts.ipynb`](03_neighborhood_preserving_layouts.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/likr/egraph-rs/blob/master/notebooks/03_neighborhood_preserving_layouts.ipynb) | **近傍保持レイアウト（Track A / Track B / 統一近傍ソルバ）**: `TsNetSolver`（`backend="auto"` / `resolved_backend`, `TwoStepRandomWalk` (`mode="two_step_random_walk"`), `stage_iterations`, `TsNetStage` / `from_probabilities_with_stages`, 既定 `interpolation_order=3`）・`SparseProbabilities.from_csr`・`RhghLandmarks` / `compute_pivoted_rhgh_landmarks`・Track B（`SgdSolver.sparse_barrier_dog`, `GuMapSolver`）・`TFdpSolver` / `IbUmapSolver`（`iterations=`, `sigma_tolerance=`） / `SgdSolver` 近傍ファミリー | **利用可能 (Ready)** |
| 04 | [`04_force_directed_and_multilevel_layouts.ipynb`](04_force_directed_and_multilevel_layouts.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/likr/egraph-rs/blob/master/notebooks/04_force_directed_and_multilevel_layouts.ipynb) | **力学モデル・マルチレベル・空間充填曲線・バックボーン**: `ForceAtlas2Solver`（`max_displacement` 等全ゲッタ・3D `LayoutNd` 対応）・`YifanHuSolver`（3D `LayoutNd` 対応）・`FruchtermanReingoldSolver`・`LinLogSolver`・`GemSolver`（`max_iterations` 上限ガード `min(4N, 400)`）・`SelLayoutSolver`（`fr_config`, `t_fdp_config`）・`SfcLayoutSolver`・`BackboneSolver`・`extract_quadrilateral_backbone`（`use_weights=True` 既定）・`coarse_backbone_init` | **利用可能 (Ready)** |
| 05 | [`05_community_clustered_and_dag_layouts.ipynb`](05_community_clustered_and_dag_layouts.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/likr/egraph-rs/blob/master/notebooks/05_community_clustered_and_dag_layouts.ipynb) | **コミュニティ抽出・クラスタレイアウト・DAG 階層描画**: ゼロコピー Result クラス 5 種（`LouvainResult`, `LeidenResult` `.modularity` vs `.quality`, `LabelPropagationResult`, `SpectralClusteringResult`, `KMeansResult`）・`coarsen_graph` / `louvain_coarsen` / `GroupInABoxSolver`（`fit_bounds=True` 既定）/ `squarified_treemap` / `ClusterRefineSolver` / `refine_cluster_layout(..., graph=...)`・`generate_layered_dag` / `cycle_edges` / `remove_cycles` / `transitive_reduction` / `assign_layers` / `project_1d`（インプレース・`None` 返却） | **利用可能 (Ready)** |
| 06 | [`06_postprocessing_overlap_and_bundling.ipynb`](06_postprocessing_overlap_and_bundling.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/likr/egraph-rs/blob/master/notebooks/06_postprocessing_overlap_and_bundling.ipynb) | **後処理・重なり除去・密度場リファイン・エッジバンドリング**: `PrismOverlapRemoval` / `PfsPrimeOverlapRemoval` / `GTreeOverlapRemoval` / `ForbidOverlapRemoval`（既定 `variant="sparse_forbid"`）・`TwinDeclutterer` / `declutter_twins_2d`・`SpatialIntruderRefiner`（`iterations=`）・`ScreenedPoissonField2d` / `ScreenedPoissonRefiner`（既定 `grid_size=512`, `density_grid()`, 4 スカラー引数 `evaluate_in_bounds`）・`FdebSolver(graph, ...)`（統一 `step(layout)`/`run(layout)`・2D/N-D 対応・`inspect.signature` 対応） | **利用可能 (Ready)** |
| 07 | [`07_non_euclidean_manifold_layouts.ipynb`](07_non_euclidean_manifold_layouts.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/likr/egraph-rs/blob/master/notebooks/07_non_euclidean_manifold_layouts.ipynb) | **非ユークリッド多様体レイアウト**: `ManifoldLayout2d`（4 種多様体: ポアンカレ円板・双曲ローレンツ `hyperbolic_lorentz`・単位球面 $S^2$・トーラス $T^2$、`.kind`/`__len__`/`coords(copy=False)`/`ManifoldMetric`）・全 17 種 `SgdSolver` と 6 種多様体ソルバの統一 `step(m)`/`run(m)`・`reset_schedule`/`run_schedule_window`・`torus_spectral_phase_init` / `torus_auto_pan` / `set_torus_extents` / `torus_layout_edge_segments`・`geodesic_distortion`・球面/双曲/トーラス交差数・`resistance_curvature_flow`（既定 `exact_max_nodes=256`） | **利用可能 (Ready)** |
| 08 | [`08_quality_metrics_and_benchmarks.ipynb`](08_quality_metrics_and_benchmarks.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/likr/egraph-rs/blob/master/notebooks/08_quality_metrics_and_benchmarks.ipynb) | **レイアウト品質評価指標とベンチマーク比較**: `evaluate_metrics` -> `MetricReport`（全 53 フィールド、`labels=`、空白除去正規化、重み付き `ideal_edge_length` スケーリング、3D `LayoutNd`/多様体対応）・`adjusted_patch_nig_detail`・`histogram_density_measure`・`ideal_edge_lengths_scaled`・`layout_degeneracy`（13 キー）・`landmark_stratified_rta_from_landmarks`（`np.inf` 直接対応）・7 アルゴリズム定量比較ベンチマーク | **利用可能 (Ready)** |
| 09 | [`09_spectral_diffusion_and_amg_primitives.ipynb`](09_spectral_diffusion_and_amg_primitives.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/likr/egraph-rs/blob/master/notebooks/09_spectral_diffusion_and_amg_primitives.ipynb) | **スペクトル拡散・チェビシェフ多項式・AMG-PCG プリミティブ**: `CsrMatrix`・組み込みラプラシアン構築 5 種（`laplacian`, `combinatorial_laplacian`, `symmetric_normalized_laplacian`, `random_walk_laplacian`, `shifted_normalized_laplacian`）・`DijkstraWorkspace` / `all_pairs_dijkstra`・`ChebyshevDogKernel` / `adaptive_chebyshev_degree` / `build_regularized_dog_terms` / `project_psd_spectral_filter`・`AmgPcgSolver`（対称ラプラシアン検証）・`compute_effective_resistances`・`resistance_curvature_flow` | **利用可能 (Ready)** |
