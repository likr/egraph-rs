import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import init, * as eg from "../pkg/egraph_wasm.node.js";
import * as webPkg from "../pkg/index.js";
import { runClusteringAndSolversSmoke } from "./esm_smoke/clustering_and_solvers.mjs";
import {
  runCoarsenAndFieldSmoke,
  runGraphBaselinesSmoke,
} from "./esm_smoke/coarsen_field_and_graph_baselines.mjs";
import { runCommunityOverlapAndClusterMetricsSmoke } from "./esm_smoke/community_overlap_and_cluster_metrics.mjs";
import { runCoreBuffersSmoke, runLegacyAbsenceSmoke } from "./esm_smoke/core_buffers.mjs";
import { runDatasetAndProgressiveSmoke } from "./esm_smoke/dataset_and_progressive.mjs";
import { runForceAndConstraintsSmoke } from "./esm_smoke/force_and_constraints.mjs";
import { runLinalgAndNeighborhoodSmoke } from "./esm_smoke/linalg_and_neighborhood.mjs";
import { runMetricsClusterAreaSmoke } from "./esm_smoke/metrics_cluster_area.mjs";
import { runMetricsLayoutAndSpatialSmoke } from "./esm_smoke/metrics_layout_and_spatial.mjs";
import { runSpatialTreesDatasetsAndNdSmoke } from "./esm_smoke/spatial_trees_datasets_and_nd.mjs";

// Cross-platform init() should resolve cleanly even when Node loader pre-initialized WASM.
await init();
eg.setPanicHook();

const ctx = { eg, webPkg };

runCoreBuffersSmoke(ctx);
runMetricsLayoutAndSpatialSmoke(ctx);
runMetricsClusterAreaSmoke(ctx);
await runLinalgAndNeighborhoodSmoke(ctx);
await runClusteringAndSolversSmoke(ctx);
await runForceAndConstraintsSmoke(ctx);
runLegacyAbsenceSmoke(ctx);
await runDatasetAndProgressiveSmoke(ctx);
runCoarsenAndFieldSmoke(ctx);
await runCommunityOverlapAndClusterMetricsSmoke(ctx);
await runGraphBaselinesSmoke(ctx);
await runSpatialTreesDatasetsAndNdSmoke(ctx);

// Run TypeScript declaration typechecking (requires local `tsc` binary)
{
  const dtsPath = fileURLToPath(new URL("../pkg/index.d.ts", import.meta.url));
  const tscBinName = process.platform === "win32" ? "tsc.cmd" : "tsc";
  const tscCandidates = [
    fileURLToPath(new URL(`../node_modules/.bin/${tscBinName}`, import.meta.url)),
    fileURLToPath(new URL(`../../../node_modules/.bin/${tscBinName}`, import.meta.url)),
  ];
  const tscBin = tscCandidates.find((p) => existsSync(p));
  if (!tscBin) {
    throw new Error(
      `Missing tsc binary for crates/wasm/pkg/index.d.ts typechecking (checked: ${tscCandidates.join(", ")})`,
    );
  }
  execFileSync(tscBin, ["--noEmit", dtsPath], { stdio: "inherit" });
}

console.log("NODE_ESM_SMOKE_OK");
