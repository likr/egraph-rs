import { useEffect, useState } from "react";
import * as d3 from "d3";
import {
  ManifoldLayout2d,
  SgdSolver,
  runProgressive,
  torusLayoutEdgeSegments,
  wrapTorusPoint,
} from "egraph";
import { loadMiserablesGraph } from "../graphUtils";

function torusShortestDelta(a, b) {
  let d = b - a;
  if (d > 0.5) d -= 1.0;
  if (d < -0.5) d += 1.0;
  return d;
}

function torusSegments(x1, y1, x2, y2, size) {
  const dx = torusShortestDelta(x1, x2);
  const dy = torusShortestDelta(y1, y2);
  const segs = [];
  for (let i = -1; i <= 1; i++) {
    for (let j = -1; j <= 1; j++) {
      const sx = (x1 + i) * size;
      const sy = (y1 + j) * size;
      const tx = (x1 + i + dx) * size;
      const ty = (y1 + j + dy) * size;
      if (
        Math.max(sx, tx) >= 0 &&
        Math.min(sx, tx) <= size &&
        Math.max(sy, ty) >= 0 &&
        Math.min(sy, ty) <= size
      ) {
        segs.push([
          [sx, sy],
          [tx, ty],
        ]);
      }
    }
  }
  return segs;
}

export function ExampleTorusGeometry() {
  const size = 1000;
  const [scene, setScene] = useState({ nodes: [], links: [] });
  const [offset, setOffset] = useState([0, 0]);
  const [status, setStatus] = useState("initializing...");

  useEffect(() => {
    const ac = new AbortController();
    let graph = null;
    let manifold = null;
    let solver = null;

    (async () => {
      try {
        const loaded = await loadMiserablesGraph();
        if (ac.signal.aborted) {
          loaded.graph.free();
          return;
        }
        graph = loaded.graph;
        manifold = ManifoldLayout2d.torus2d(loaded.numNodes);
        for (let i = 0; i < loaded.numNodes; i++) {
          const [wx, wy] = wrapTorusPoint((i * 0.6180339) % 1.0, (i * 0.381966) % 1.0);
          manifold.setPoint(i, wx, wy);
        }
        solver = SgdSolver.full(graph, { iterations: 40, eps: 0.1, seed: 1 });

        const syncFrame = (ctx, targetLayout = manifold, segCount = null) => {
          const raw = targetLayout.coordsView();
          const nodes = loaded.nodes.map((node, i) => {
            const [wx, wy] = wrapTorusPoint(raw[2 * i], raw[2 * i + 1]);
            return { ...node, tx: wx, ty: wy };
          });
          setScene({ nodes, links: loaded.links });
          const segSuffix = segCount !== null ? ` | ${segCount} wrapped segments` : "";
          setStatus(`Torus 2D SGD: ${ctx.iteration}/${ctx.totalIterations} (${ctx.status})${segSuffix}`);
        };

        const finalCtx = await runProgressive(solver, manifold, (ctx) => syncFrame(ctx), {
          frameBudgetMs: 10,
          signal: ac.signal,
        });
        if (!ac.signal.aborted) {
          const panned = manifold.torusAutoPan(graph);
          const segs = torusLayoutEdgeSegments(graph, panned);
          syncFrame(finalCtx, panned, segs.length / 4);
          panned.free();
        }
      } catch (err) {
        if (err?.name !== "AbortError") {
          setStatus(`Error: ${err.message}`);
        }
      }
    })();

    return () => {
      ac.abort();
      solver?.free();
      manifold?.free();
      graph?.free();
    };
  }, []);

  const tiles = [-1, 0, 1];
  return (
    <div>
      <div className="graph-status-bar">
        <span className="tag is-info is-medium">Torus 2D Geometry (ManifoldLayout2d + runProgressive)</span>
        <span className="tag is-light">{status}</span>
        <span className="tag is-white">Click a node to shift torus origin</span>
      </div>
      <div className="graph-canvas-wrap">
        <svg viewBox={`0 0 ${size} ${size}`}>
          <g>
            {scene.links.map((link) => {
              const s = scene.nodes[link.source];
              const t = scene.nodes[link.target];
              if (!s || !t) return null;
              const sx = (((s.tx - offset[0]) % 1) + 1) % 1;
              const sy = (((s.ty - offset[1]) % 1) + 1) % 1;
              const tx = (((t.tx - offset[0]) % 1) + 1) % 1;
              const ty = (((t.ty - offset[1]) % 1) + 1) % 1;
              const path = d3.path();
              for (const [[x1, y1], [x2, y2]] of torusSegments(sx, sy, tx, ty, size)) {
                path.moveTo(x1, y1);
                path.lineTo(x2, y2);
              }
              return (
                <path
                  key={`${link.source}:${link.target}`}
                  d={path.toString()}
                  fill="none"
                  stroke="#999"
                  strokeWidth="1.2"
                  opacity="0.55"
                />
              );
            })}
            {scene.nodes.map((node) => {
              const nx = (((node.tx - offset[0]) % 1) + 1) % 1;
              const ny = (((node.ty - offset[1]) % 1) + 1) % 1;
              return tiles.flatMap((i) =>
                tiles.map((j) => {
                  const px = (nx + i) * size;
                  const py = (ny + j) * size;
                  if (px < -20 || px > size + 20 || py < -20 || py > size + 20) {
                    return null;
                  }
                  return (
                    <circle
                      key={`${node.id}:${i}:${j}`}
                      className="is-clickable"
                      cx={px}
                      cy={py}
                      r="8"
                      fill={node.color}
                      stroke="#fff"
                      strokeWidth="1.5"
                      onClick={() => setOffset([node.tx - 0.5, node.ty - 0.5])}
                    />
                  );
                }),
              );
            })}
          </g>
        </svg>
      </div>
    </div>
  );
}
