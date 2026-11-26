# `egraph` v6 Python チュートリアル（Google Colab 対応）

`egraph` v6 は、**Pure Rust による高速・決定論的・シングルスレッド**なグラフレイアウト計算および品質評価ライブラリです。

> [!IMPORTANT]
> **役割の分離（レイアウト計算エンジンとしての `egraph`）**  
> `egraph` v6 は **グラフデータ構造（`CsrGraph`）・レイアウト計算（`Layout2d` / `LayoutNd` / `ManifoldLayout2d` / 各種ソルバ）・コミュニティ抽出／階層化・品質評価指標（`evaluate_metrics` など）のみ** を提供します。  
> ネットワーク図の描画・可視化はすべて Python の標準的な可視化ライブラリ（`matplotlib`・`networkx`・`numpy`）を組み合わせて行います。

---

## Google Colab / ローカル環境でのセットアップ

- **動作要件**: Python `>= 3.10`、`numpy >= 1.26`
- **Google Colab** では各ノートブックの先頭セルで次のインストールコマンドを実行してください（現在 v6 はプレリリース版のため `--pre` が必要です）。

### 標準インストール（NB02〜NB09）

```bash
pip install -q --pre "egraph==6.0.0a8" matplotlib-fontja
```

### 相互運用・可視化拡張付きインストール（NB01 推奨）

`NetworkX` / `SciPy` / `igraph` / `matplotlib` との相互運用をすべて試す場合は `[interop,viz]` extra を指定します。

```bash
pip install -q --pre "egraph[interop,viz]==6.0.0a8" matplotlib-fontja
```

- `matplotlib-fontja` は Google Colab 上の Matplotlib で日本語ラベルを文字化けなく表示するためのパッケージです（`import matplotlib_fontja` で自動適用されます）。ローカル環境等で未インストールの場合は `try: import matplotlib_fontja; except ImportError: pass` によりフォールバックできます。
- Google Colab 上で全セルをまとめて実行する場合は、「ランタイム」→「すべてのセルを実行」を選択してください。

---

## v5 から v6 への移行・関連ドキュメント

`egraph` v6 では従来の `Graph` / `DrawingEuclidean2d` / `TsNet` などから、`CsrGraph` / `Layout2d` / 統一ソルバ API への破壊的変更が行われています。v5 からの移行手順と API 対応表は以下のドキュメントを参照してください。

- **v5 → v6 移行ガイド**: [`../docs/MIGRATION_V6.md`](../docs/MIGRATION_V6.md)
- **Python API ドキュメント（Sphinx）**: [https://egraph-py-docs.netlify.app/](https://egraph-py-docs.netlify.app/)

---

## ノートブック一覧（全 9 本ロードマップ）

トピックごとに以下の 9 本のチュートリアルノートブックで構成されています。

| # | ファイル名 | タイトル・扱う主なトピック | 状態 |
|---|---|---|---|
| 01 | [`01_getting_started_and_data_structures.ipynb`](01_getting_started_and_data_structures.ipynb) | **入門とデータ構造**: `CsrGraph`・`load_dataset`（全 9 データセット）・`parse_csr_csv`・`Layout2d` / `LayoutNd` ゼロコピーバッファ・`egraph.interop`（NetworkX / SciPy / igraph）・空間補助機能（Delaunay・矩形重なり判定）・`matplotlib` 可視化ヘルパー | **利用可能 (Ready)** |
| 02 | [`02_initialization_and_stress_layouts.ipynb`](02_initialization_and_stress_layouts.ipynb) | **初期配置とストレス系レイアウト**: `pivot_mds_init` / `koren_spectral_init` / `classical_mds_2d` / `hde` / `hde_nd` / `rd_mds` / `representative_mds`・3D `LayoutNd` 可視化・`StressMajorizationSolver` / `KamadaKawaiSolver` / `MaxentStressSolver` / `GripSolver`・`SgdSolver` ストレス系と `step()` / `run_steps()` 協調実行 | **利用可能 (Ready)** |
| 03 | [`03_neighborhood_preserving_layouts.ipynb`](03_neighborhood_preserving_layouts.ipynb) | **近傍保持レイアウト（Track A / Track B / 統一近傍ソルバ）**: `TsNetSolver`（全 5 コンストラクタ・4 バックエンド）・`SgdSolver.sparse_barrier_dog` / `GuMapSolver`・`TFdpSolver` / `IbUmapSolver` / `snap_t_fdp` / `pacmap_graph*` / `largevis` / `drgraph` / `graph_cne`・近傍／パープレキシティ／DoG・rHGH プリミティブ全 26 種 | **利用可能 (Ready)** |
| 04 | `04_force_directed_and_multilevel_layouts.ipynb` | **力学モデル・マルチレベル・空間充填曲線・バックボーン**: `ForceAtlas2Solver`・`YifanHuSolver`・`FruchtermanReingoldSolver`・`LinLogSolver`・`GemSolver`・`SelLayoutSolver`・`SfcLayoutSolver`・`BackboneSolver`・`extract_quadrilateral_backbone`・`coarse_backbone_init` | 作成予定 (Planned) |
| 05 | `05_community_clustered_and_dag_layouts.ipynb` | **コミュニティ抽出・クラスタレイアウト・DAG 階層描画**: `louvain_communities` / `leiden_communities` / `label_propagation` / `spectral_clustering`・`coarsen_graph` / `GroupInABoxSolver` / `squarified_treemap` / `ClusterRefineSolver`・`cycle_edges` / `assign_layers` / `SeparationConstraint` | 作成予定 (Planned) |
| 06 | `06_postprocessing_overlap_and_bundling.ipynb` | **後処理・重なり除去・エッジバンドリング**: `PrismOverlapRemoval` / `PfsPrimeOverlapRemoval` / `GTreeOverlapRemoval` / `ForbidOverlapRemoval`・`SpatialIntruderRefiner`・`ScreenedPoissonField2d` / `ScreenedPoissonRefiner`・`FdebSolver` | 作成予定 (Planned) |
| 07 | `07_non_euclidean_manifold_layouts.ipynb` | **非ユークリッド多様体レイアウト**: `ManifoldLayout2d`（双曲ポアンカレ円板・球面・トーラス）・`SgdSolver.run_manifold`・メビウス変換／測地線距離・トーラス交差数・`resistance_curvature_flow` | 作成予定 (Planned) |
| 08 | `08_quality_metrics_and_benchmarks.ipynb` | **レイアウト品質評価指標とベンチマーク比較**: `evaluate_metrics`（全 42 指標）・ストレス／BFS／交差数／角度／幾何／近傍保持／RTA／クラスタ／NOR 単体指標・複数アルゴリズムの定量比較 | 作成予定 (Planned) |
| 09 | `09_spectral_diffusion_and_amg_primitives.ipynb` | **スペクトル拡散・チェビシェフ多項式・AMG-PCG プリミティブ**: `CsrMatrix`・`anisotropic_laplacian`・`smallest_laplacian_eigenpairs`・`chebyshev_heat_spmv*`・`chebyshev_dog_jl_embedding*`・`AmgPcgSolver`・`compute_effective_resistances` | 作成予定 (Planned) |
