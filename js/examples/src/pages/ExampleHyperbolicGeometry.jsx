import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { ManifoldLayout2d, SgdSolver, runProgressive } from "egraph";
import { createManifoldRunner, loadMiserablesGraph } from "../graphUtils";

function recenter(cx, cy, x, y) {
  const dx = x - cx;
  const dy = y - cy;
  const dr = 1 - cx * x - cy * y;
  const di = cy * x - cx * y;
  const d = dr * dr + di * di;
  return [(dr * dx + di * dy) / d, (dr * dy - di * dx) / d];
}

function projectLinks(nodes, links, displayR) {
  for (const link of links) {
    const { x: x1, y: y1 } = nodes[link.source];
    const { x: x2, y: y2 } = nodes[link.target];
    const b1 = (x1 * x1 + y1 * y1 + displayR * displayR) / 2;
    const b2 = (x2 * x2 + y2 * y2 + displayR * displayR) / 2;
    const d = x1 * y2 - y1 * x2;
    if (Math.abs(d) > 1e-6) {
      const cx = (link.cx = (y2 * b1 - y1 * b2) / d);
      const cy = (link.cy = (x1 * b2 - x2 * b1) / d);
      const radSq = cx * cx + cy * cy - displayR * displayR;
      link.r = radSq > 0 ? Math.sqrt(radSq) : 0;
    } else {
      link.r = 0;
    }
    link.x1 = x1;
    link.y1 = y1;
    link.x2 = x2;
    link.y2 = y2;
  }
}

export function ExampleHyperbolicGeometry() {
  const svgRef = useRef();
  const [scene, setScene] = useState({ nodes: [], links: [] });
  const [status, setStatus] = useState("initializing...");
  const displayR = 500;
  const margin = 50;

  useEffect(() => {
    const drag = d3.drag().on("drag", (event) => {
      const cx = -event.dx / displayR;
      const cy = -event.dy / displayR;
      setScene((prev) => {
        const nodes = prev.nodes.map((node) => {
          const [x, y] = recenter(cx, cy, node.x / displayR, node.y / displayR);
          return { ...node, x: x * displayR, y: y * displayR };
        });
        const links = prev.links.map((l) => ({ ...l }));
        projectLinks(nodes, links, displayR);
        return { nodes, links };
      });
    });
    d3.select(svgRef.current).call(drag);
  }, []);

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
        manifold = ManifoldLayout2d.hyperbolicPoincare(loaded.numNodes);
        for (let i = 0; i < loaded.numNodes; i++) {
          const ang = (2 * Math.PI * i) / loaded.numNodes;
          const r = 0.08 * (1 + (i % 4)) * 0.25;
          manifold.setPoint(i, r * Math.cos(ang), r * Math.sin(ang));
        }
        solver = SgdSolver.full(graph, { tMax: 80, eps: 0.1, seed: 4 });
        const runner = createManifoldRunner(solver);

        const syncFrame = (ctx) => {
          const raw = manifold.coords();
          const nodes = loaded.nodes.map((node, i) => ({
            ...node,
            x: raw[2 * i] * displayR,
            y: raw[2 * i + 1] * displayR,
          }));
          const links = loaded.links.map((l) => ({ ...l }));
          projectLinks(nodes, links, displayR);
          setScene({ nodes, links });
          setStatus(`Hyperbolic Poincaré SGD: ${ctx.iteration}/${ctx.totalIterations} (${ctx.status})`);
        };

        await runProgressive(runner, manifold, syncFrame, {
          frameBudgetMs: 10,
          signal: ac.signal,
        });
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

  const size = (displayR + margin) * 2;
  return (
    <div>
      <div className="graph-status-bar">
        <span className="tag is-info is-medium">Hyperbolic Geometry (ManifoldLayout2d + runProgressive)</span>
        <span className="tag is-light">{status}</span>
        <span className="tag is-white">Drag disk to recenter</span>
      </div>
      <div className="graph-canvas-wrap">
        <svg ref={svgRef} style={{ cursor: "move" }} viewBox={`${-margin} ${-margin} ${size} ${size}`}>
          <defs>
            {scene.links.map((link) => {
              const path = d3.path();
              path.moveTo(0, 0);
              path.lineTo(link.x1, link.y1);
              path.lineTo(link.x2, link.y2);
              path.closePath();
              return (
                <clipPath key={`${link.source}:${link.target}`} id={`clip:${link.source}:${link.target}`}>
                  <path d={path.toString()} stroke="#000" strokeWidth={link.strokeWidth} />
                </clipPath>
              );
            })}
          </defs>
          <g transform={`translate(${displayR},${displayR})`}>
            <circle r={displayR} fill="none" stroke="#888" />
            <g>
              {scene.links.map((link) =>
                link.r > 0 ? (
                  <circle
                    key={`${link.source}:${link.target}`}
                    cx={link.cx}
                    cy={link.cy}
                    r={link.r}
                    fill="none"
                    stroke="#999"
                    strokeWidth={link.strokeWidth}
                    opacity="0.55"
                    clipPath={`url(#clip:${link.source}:${link.target})`}
                  />
                ) : (
                  <line
                    key={`${link.source}:${link.target}`}
                    x1={link.x1}
                    y1={link.y1}
                    x2={link.x2}
                    y2={link.y2}
                    stroke="#999"
                    strokeWidth={link.strokeWidth}
                    opacity="0.55"
                  />
                ),
              )}
              {scene.nodes.map((node) => (
                <g key={node.id} transform={`translate(${node.x},${node.y})`}>
                  <circle r="5" fill={node.color} stroke="#fff" strokeWidth="1.5" />
                  <text className="is-unselectable" fontSize="9" dx="7" dy="3">
                    {node.name}
                  </text>
                </g>
              ))}
            </g>
          </g>
        </svg>
      </div>
    </div>
  );
}
