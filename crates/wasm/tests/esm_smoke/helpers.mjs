import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

export function assertJsError(fn, label) {
  let threw = false;
  try {
    fn();
  } catch (err) {
    assert(!(err instanceof WebAssembly.RuntimeError), `${label} trapped with RuntimeError`);
    assert(err instanceof Error, `${label} must throw Error`);
    threw = true;
  }
  assert(threw, `${label} must throw`);
}

export function readPkgText(name) {
  return readFile(new URL(`../../pkg/${name}`, import.meta.url), "utf8");
}

export function readKarateCsv() {
  return readFile(new URL("../../../dataset/src/data/karate.csv", import.meta.url), "utf8");
}

export function makeBarbellGraph(eg, k) {
  const edges = [];
  for (const off of [0, k]) {
    for (let i = 0; i < k; i++) {
      for (let j = i + 1; j < k; j++) {
        edges.push(off + i, off + j);
      }
    }
  }
  edges.push(k - 1, k);
  return eg.CsrGraph.fromEdgeList(k * 2, new Uint32Array(edges), null, false, true);
}

export function makeGridGraph(eg, rows, cols) {
  const edges = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const u = r * cols + c;
      if (c + 1 < cols) edges.push(u, u + 1);
      if (r + 1 < rows) edges.push(u, u + cols);
    }
  }
  return eg.CsrGraph.fromEdgeList(rows * cols, new Uint32Array(edges), null, false, true);
}

export function makeCircleCoords(n, radius = 1.0) {
  const out = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    const ang = (2 * Math.PI * i) / Math.max(n, 1);
    out[2 * i] = radius * Math.cos(ang);
    out[2 * i + 1] = radius * Math.sin(ang);
  }
  return out;
}
