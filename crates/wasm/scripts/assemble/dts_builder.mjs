import { applyCheckedReplacements, applyCoreAndMetricsDts } from "./dts_core_metrics.mjs";
import {
  clusterBaselinesDtsReplacements,
  multilevelStressBaselinesDtsReplacements,
} from "./dts_baselines.mjs";
import {
  coarseningFieldDtsReplacements,
  forceOverlapDtsReplacements,
} from "./dts_force_field.mjs";
import {
  clusteringSolversDtsReplacements,
  linalgNeighborhoodDtsReplacements,
} from "./dts_linalg_solvers.mjs";
import { polyMetricsAndSolverDeclarations } from "./dts_poly_decls.mjs";
import { clusterFieldTypeDecls } from "./dts_types_cluster_field.mjs";
import { layoutBaselineAndProgressiveTypeDecls } from "./dts_types_layout_baselines.mjs";
import { solverEnumAndPrimaryTypeDecls } from "./dts_types_solvers.mjs";

export function buildIndexDts(wasmDts, wasmDtsPath) {
  let indexDts = applyCoreAndMetricsDts(wasmDts, wasmDtsPath);

  indexDts = applyCheckedReplacements(
    indexDts,
    linalgNeighborhoodDtsReplacements,
    "linear-algebra/neighborhood",
    wasmDtsPath,
  );

  if (!indexDts.includes("export class CoarsenedGraphResult {")) {
    throw new Error(`Missing CoarsenedGraphResult declaration in ${wasmDtsPath}`);
  }
  const combinedTypeDecls = `${solverEnumAndPrimaryTypeDecls}\n${clusterFieldTypeDecls}\n${layoutBaselineAndProgressiveTypeDecls}`;
  indexDts = indexDts.replace(
    "export class CoarsenedGraphResult {",
    `${combinedTypeDecls}export class CoarsenedGraphResult {`,
  );

  indexDts = applyCheckedReplacements(
    indexDts,
    clusteringSolversDtsReplacements,
    "clustering/solver",
    wasmDtsPath,
  );
  indexDts = applyCheckedReplacements(
    indexDts,
    forceOverlapDtsReplacements,
    "force/overlap/constraint",
    wasmDtsPath,
  );
  indexDts = applyCheckedReplacements(
    indexDts,
    coarseningFieldDtsReplacements,
    "coarsening/field",
    wasmDtsPath,
  );
  indexDts = applyCheckedReplacements(
    indexDts,
    clusterBaselinesDtsReplacements,
    "cluster/overlap baseline",
    wasmDtsPath,
  );
  indexDts = applyCheckedReplacements(
    indexDts,
    multilevelStressBaselinesDtsReplacements,
    "multilevel/stress baseline",
    wasmDtsPath,
  );

  return `${indexDts}\n${polyMetricsAndSolverDeclarations}`;
}
