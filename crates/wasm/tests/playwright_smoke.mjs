import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import http from "node:http";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "../../..");
const pkgDir = path.join(repoRoot, "crates/wasm/pkg");
const distDir = path.join(repoRoot, "js/examples/dist");

async function loadPlaywright() {
  try {
    return await import("playwright");
  } catch {
    const fallback = path.join(repoRoot, "js/examples/node_modules/playwright/index.mjs");
    return await import(pathToFileURL(fallback).href);
  }
}

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".mjs": "application/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".svg": "image/svg+xml",
};

function startStaticServer() {
  const server = http.createServer(async (req, res) => {
    try {
      const reqUrl = new URL(req.url, "http://127.0.0.1");
      const pathname = decodeURIComponent(reqUrl.pathname);

      let filePath = null;
      if (pathname.startsWith("/pkg/")) {
        filePath = path.join(pkgDir, pathname.slice("/pkg/".length));
      } else {
        const candidate = path.join(distDir, pathname === "/" ? "index.html" : pathname.slice(1));
        if (existsSync(candidate) && !candidate.endsWith(path.sep)) {
          filePath = candidate;
        } else {
          // SPA fallback for Vite client-side routes
          filePath = path.join(distDir, "index.html");
        }
      }

      const data = await readFile(filePath);
      const ext = path.extname(filePath);
      res.writeHead(200, { "Content-Type": MIME_TYPES[ext] ?? "application/octet-stream" });
      res.end(data);
    } catch (err) {
      res.writeHead(404);
      res.end(`Not found: ${err.message}`);
    }
  });

  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address();
      resolve({ server, baseUrl: `http://127.0.0.1:${addr.port}` });
    });
  });
}

async function launchBrowser(chromium) {
  const chromePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
  try {
    return await chromium.launch({ headless: true });
  } catch {
    return await chromium.launch({
      headless: true,
      executablePath: chromePath,
    });
  }
}

const { chromium } = await loadPlaywright();
const { server, baseUrl } = await startStaticServer();
const browser = await launchBrowser(chromium);

try {
  const page = await browser.newPage();
  const pageErrors = [];
  const consoleErrors = [];
  page.on("pageerror", (err) => pageErrors.push(err.message));
  page.on("console", (msg) => {
    if (msg.type() === "error") {
      consoleErrors.push(msg.text());
    }
  });

  // 1 & 2. Load crates/wasm/pkg/index.js directly in Chromium, verify memory.grow detach resilience,
  // and measure N=5,000 runProgressive per-frame work times against <= 10 ms/frame budget.
  await page.goto(`${baseUrl}/`, { waitUntil: "domcontentloaded" });

  const browserBench = await page.evaluate(async (pkgUrl) => {
    const eg = await import(pkgUrl);
    await eg.default();

    // Test 1: Layout2d & LayoutNd coordsView across memory.grow
    const l2 = eg.Layout2d.fromCoords(new Float32Array([1.5, -2.5, 3.25, 4.75]));
    const lnd = eg.LayoutNd.fromCoords(2, 3, new Float32Array([1, 2, 3, 4, 5, 6]));
    const v2Before = l2.coordsView();
    const vndBefore = lnd.coordsView();
    const bufBefore2 = v2Before.buffer;
    const bufBeforeNd = vndBefore.buffer;

    eg.wasmMemory().grow(2);

    const detached2Len = bufBefore2.byteLength;
    const detachedNdLen = bufBeforeNd.byteLength;
    const v2After = l2.coordsView();
    const vndAfter = lnd.coordsView();
    const l2Ok =
      detached2Len === 0 &&
      v2After.buffer.byteLength > 0 &&
      v2After[0] === 1.5 &&
      v2After[3] === 4.75;
    const lndOk =
      detachedNdLen === 0 &&
      vndAfter.buffer.byteLength > 0 &&
      vndAfter[0] === 1 &&
      vndAfter[5] === 6;

    l2.free();
    lnd.free();

    // Test 2: N=5,000 (50x100 grid) runProgressive <= 10 ms/frame budget
    const rows = 50;
    const cols = 100;
    const n = rows * cols;
    const edges = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const u = r * cols + c;
        if (c + 1 < cols) edges.push(u, u + 1);
        if (r + 1 < rows) edges.push(u, u + cols);
      }
    }
    const graph = eg.CsrGraph.fromEdgeList(n, new Uint32Array(edges), null, false, true);
    const layout = eg.Layout2d.init(n, 42);
    const solver = eg.SgdSolver.randomPair(graph, { pairsPerNode: 2, tMax: 24, seed: 42 });

    const frameWorkMs = [];
    let lastStamp = performance.now();
    // Wrap solver to measure pure WASM step work inside each rAF frame cleanly
    const timedSolver = {
      get isFinished() {
        return solver.isFinished;
      },
      get iteration() {
        return solver.iteration;
      },
      get totalIterations() {
        return solver.totalIterations;
      },
      get status() {
        return solver.status;
      },
      step(lay) {
        return solver.step(lay);
      },
    };

    const totalStart = performance.now();
    await eg.runProgressive(
      timedSolver,
      layout,
      () => {
        // Measure only JS+WASM slice from frame callback invocation? Wait: wrap via rAF timing
        frameWorkMs.push(performance.now() - lastStamp);
      },
      { frameBudgetMs: 6, maxStepsPerFrame: 8, stepsPerYield: 1 },
    );
    const totalMs = performance.now() - totalStart;

    // Better: direct frame-step timing inside custom wrapper so rAF vsync interval (~16.6ms)
    // is separated from actual JS/WASM main-thread blocking duration per frame!
    const layout2 = eg.Layout2d.init(n, 42);
    const solver2 = eg.SgdSolver.randomPair(graph, { pairsPerNode: 2, tMax: 24, seed: 42 });
    const pureFrameTimes = [];
    let frameStart = 0;
    let activeFrame = false;
    const instrumentedSolver = {
      get isFinished() {
        return solver2.isFinished;
      },
      get iteration() {
        return solver2.iteration;
      },
      get totalIterations() {
        return solver2.totalIterations;
      },
      get status() {
        return solver2.status;
      },
      step(lay) {
        if (!activeFrame) {
          frameStart = performance.now();
          activeFrame = true;
        }
        return solver2.step(lay);
      },
    };
    const finalCtx = await eg.runProgressive(
      instrumentedSolver,
      layout2,
      () => {
        if (activeFrame) {
          pureFrameTimes.push(performance.now() - frameStart);
          activeFrame = false;
        }
      },
      { frameBudgetMs: 6, maxStepsPerFrame: 16, stepsPerYield: 1 },
    );

    const view = layout2.coordsView();
    const coordsFinite = layout2.isAllFinite() && view.length === n * 2;
    solver2.free();
    layout2.free();
    solver.free();
    layout.free();
    graph.free();

    const sorted = [...pureFrameTimes].sort((a, b) => a - b);
    const maxMs = sorted.length > 0 ? sorted[sorted.length - 1] : 0;
    const p95Ms = sorted.length > 0 ? sorted[Math.floor(0.95 * (sorted.length - 1))] : 0;
    const medianMs = sorted.length > 0 ? sorted[Math.floor(sorted.length / 2)] : 0;

    return {
      l2Ok,
      lndOk,
      n,
      totalMs,
      frames: pureFrameTimes.length,
      medianMs,
      p95Ms,
      maxMs,
      coordsFinite,
      finalStatus: finalCtx.status,
    };
  }, `${baseUrl}/pkg/index.js`);

  assert(browserBench.l2Ok, "Layout2d.coordsView() failed memory.grow detach recovery in Chromium");
  assert(browserBench.lndOk, "LayoutNd.coordsView() failed memory.grow detach recovery in Chromium");
  assert(browserBench.coordsFinite, "N=5,000 layout produced non-finite coordinates");
  assert.equal(browserBench.finalStatus, "completed");
  assert(
    browserBench.maxMs <= 10.0,
    `Expected N=5,000 runProgressive frame time <= 10 ms, got max=${browserBench.maxMs.toFixed(2)} ms`,
  );

  // 3. Load built js/examples/dist/ SPA and verify all 13 demo routes render cleanly without errors
  const canvasRoutes = [
    "/tsnet-track-a",
    "/t-fdp",
    "/force-atlas2",
    "/gumap",
    "/sgd",
    "/kamada-kawai",
    "/stress-majorization",
    "/mds",
    "/edge-bundling",
    "/prism-overlap-removal",
  ];
  const svgRoutes = [
    "/hyperbolic-geometry",
    "/spherical-geometry",
    "/torus-geometry",
  ];
  await page.goto(`${baseUrl}/`, { waitUntil: "networkidle" });
  await page.waitForSelector("a[href='/tsnet-track-a']");

  for (const route of canvasRoutes) {
    await page.goto(`${baseUrl}${route}`, { waitUntil: "networkidle" });
    await page.waitForSelector("[data-testid='graph-canvas']");
    await page.waitForTimeout(250);
  }

  for (const route of svgRoutes) {
    await page.goto(`${baseUrl}${route}`, { waitUntil: "networkidle" });
    await page.waitForSelector(".graph-canvas-wrap svg > g circle, .graph-canvas-wrap svg > g path");
    await page.waitForTimeout(250);
  }

  assert.equal(pageErrors.length, 0, `Page errors in Chromium: ${pageErrors.join("; ")}`);
  assert.equal(consoleErrors.length, 0, `Console errors in Chromium: ${consoleErrors.join("; ")}`);

  console.log(
    `PLAYWRIGHT_SMOKE_OK: n=${browserBench.n}, frames=${browserBench.frames}, medianMs=${browserBench.medianMs.toFixed(3)}, p95Ms=${browserBench.p95Ms.toFixed(3)}, maxMs=${browserBench.maxMs.toFixed(3)}`,
  );
} finally {
  await browser.close();
  server.close();
}
