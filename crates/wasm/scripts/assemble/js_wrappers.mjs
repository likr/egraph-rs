import { polyExportDefinitions, polyWrapperRuntimeHelpers } from "./js_poly_wrappers.mjs";

const WRAPPED_CLASSES = new Set([
  "Layout2d",
  "Layout3d",
  "LayoutNd",
  "ManifoldLayout2d",
  "EdgePolylines",
  "LouvainResult",
  "LeidenResult",
  "LabelPropagationResult",
  "SpectralClusteringResult",
  "KMeansResult",
  "SgdSolver",
  "Sgd2Solver",
  "PacmapGraphSolver",
  "LargeVisSolver",
  "DrGraphSolver",
  "GraphCneSolver",
  "SnapTFdpSolver",
  "TsNetSolver",
  "GuMapSolver",
  "TFdpSolver",
  "StressMajorizationSolver",
  "KamadaKawaiSolver",
  "LinLogSolver",
  "FruchtermanReingoldSolver",
  "ForceAtlas2Solver",
  "YifanHuSolver",
  "MaxentStressSolver",
  "ArenaOctree3d",
  "DynamicKdTree",
]);

export function buildIndexJs(exportedNames) {
  const aliasedImports = [];
  const passthroughExports = [];

  for (const name of Array.from(exportedNames).sort()) {
    if (WRAPPED_CLASSES.has(name)) {
      aliasedImports.push(`  ${name} as _${name},`);
    } else {
      aliasedImports.push(`  ${name},`);
      if (!name.startsWith("_")) {
        passthroughExports.push(name);
      }
    }
  }

  const indexJs = `/* @ts-self-types="./index.d.ts" */
import init, {
${aliasedImports.join("\n")}
} from "./egraph_wasm.js";

${polyWrapperRuntimeHelpers}
/**
 * Runs a cooperative iterative solver progressively within a per-frame time budget,
 * yielding back to the browser event loop via \`requestAnimationFrame\` (or \`setTimeout\`
 * in Node.js) between batches.
 *
 * Accepts \`Layout2d\`, \`Layout3d\`, \`LayoutNd\`, \`ManifoldLayout2d\`, or \`EdgePolylines\`
 * (for \`FdebSolver\`) polymorphically.
 */
export function runProgressive(solver, layout, onFrame, options = {}) {
  const {
    frameBudgetMs = 10,
    maxStepsPerFrame = 64,
    stepsPerYield = 1,
    signal,
  } = options ?? {};

  const scheduleNext =
    typeof globalThis.requestAnimationFrame === "function"
      ? (cb) => globalThis.requestAnimationFrame(() => cb())
      : (cb) => globalThis.setTimeout(cb, 0);
  const cancelNext =
    typeof globalThis.cancelAnimationFrame === "function"
      ? (id) => globalThis.cancelAnimationFrame(id)
      : (id) => globalThis.clearTimeout(id);

  const makeAbortError = () => {
    if (typeof globalThis.DOMException === "function") {
      return new globalThis.DOMException("runProgressive aborted", "AbortError");
    }
    const err = new Error("runProgressive aborted");
    err.name = "AbortError";
    return err;
  };

  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason ?? makeAbortError());
      return;
    }
    if (solver.isFinished) {
      const doneCtx = {
        iteration: solver.iteration,
        totalIterations: solver.totalIterations,
        status: solver.status,
        layout,
      };
      if (typeof onFrame === "function") {
        onFrame(doneCtx);
      }
      resolve(doneCtx);
      return;
    }

    let handle = null;
    let settled = false;

    const onAbort = () => {
      if (settled) return;
      settled = true;
      if (handle !== null) {
        cancelNext(handle);
        handle = null;
      }
      reject(signal?.reason ?? makeAbortError());
    };

    if (signal && typeof signal.addEventListener === "function") {
      signal.addEventListener("abort", onAbort, { once: true });
    }

    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      if (signal && typeof signal.removeEventListener === "function") {
        signal.removeEventListener("abort", onAbort);
      }
      fn(value);
    };

    const tick = () => {
      if (settled) return;
      if (signal?.aborted) {
        finish(reject, signal.reason ?? makeAbortError());
        return;
      }
      try {
        const t0 = globalThis.performance.now();
        let stepsThisFrame = 0;
        let status = solver.status;
        const batch = Math.max(1, Math.floor(stepsPerYield));

        while (!solver.isFinished && stepsThisFrame < maxStepsPerFrame) {
          if (batch > 1 && typeof solver.runSteps === "function") {
            status = solver.runSteps(layout, batch);
            stepsThisFrame += batch;
          } else {
            status = solver.step(layout);
            stepsThisFrame += 1;
          }
          if (globalThis.performance.now() - t0 >= frameBudgetMs) {
            break;
          }
        }

        const ctx = {
          iteration: solver.iteration,
          totalIterations: solver.totalIterations,
          status,
          layout,
        };

        if (typeof onFrame === "function") {
          onFrame(ctx);
        }

        if (solver.isFinished) {
          finish(resolve, ctx);
          return;
        }

        handle = scheduleNext(tick);
      } catch (err) {
        finish(reject, err);
      }
    };

    handle = scheduleNext(tick);
  });
}

${polyExportDefinitions}
export {
${passthroughExports.map((name) => `  ${name},`).join("\n")}
};
export default init;
`;

  return { indexJs, exportCount: passthroughExports.length + WRAPPED_CLASSES.size + 36 };
}

export function buildNodeWrapperJs() {
  return `/* @ts-self-types="./index.d.ts" */
import { readFile } from "node:fs/promises";
import { initSync } from "./egraph_wasm.js";

const wasmBytes = await readFile(new URL("./egraph_wasm_bg.wasm", import.meta.url));
const wasmExports = initSync({ module: wasmBytes });

export * from "./index.js";

export default async function init() {
  return wasmExports;
}
`;
}
