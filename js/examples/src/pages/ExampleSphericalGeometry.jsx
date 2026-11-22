import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { ManifoldLayout2d, SgdSolver, runProgressive } from "egraph";
import { createManifoldRunner, loadMiserablesGraph } from "../graphUtils";

export function ExampleSphericalGeometry() {
  const svgRef = useRef();
  const [rotate, setRotate] = useState([0, 0]);
  const [scene, setScene] = useState({ nodes: [], links: [] });
  const [status, setStatus] = useState("initializing...");
  const displayR = 500;
  const margin = 50;

  useEffect(() => {
    const drag = d3.drag().on("drag", (event) => {
      setRotate((prev) => [prev[0] - event.dx * 0.5, prev[1] - event.dy * 0.5]);
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
        manifold = ManifoldLayout2d.spherical(loaded.numNodes);
        for (let i = 0; i < loaded.numNodes; i++) {
          const lon = (2 * Math.PI * i) / loaded.numNodes;
          const colat = Math.PI * (0.25 + 0.5 * ((i % 7) / 7));
          manifold.setPoint(i, lon, colat);
        }
        solver = SgdSolver.full(graph, { tMax: 80, eps: 0.1, seed: 4 });
        const runner = createManifoldRunner(solver);

        const syncFrame = (ctx) => {
          const raw = manifold.coords();
          const nodes = loaded.nodes.map((node, i) => ({
            ...node,
            lon: (180 * (raw[2 * i] - Math.PI)) / Math.PI,
            lat: (180 * (Math.PI / 2 - raw[2 * i + 1])) / Math.PI,
          }));
          setScene({ nodes, links: loaded.links });
          setStatus(`Spherical SGD: ${ctx.iteration}/${ctx.totalIterations} (${ctx.status})`);
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
  const projection = d3.geoAzimuthalEquidistant().rotate(rotate).translate([displayR, displayR]).scale(155);
  const path = d3.geoPath(projection);

  return (
    <div>
      <div className="graph-status-bar">
        <span className="tag is-info is-medium">Spherical Geometry (ManifoldLayout2d + runProgressive)</span>
        <span className="tag is-light">{status}</span>
        <span className="tag is-white">Drag globe to rotate</span>
      </div>
      <div className="graph-canvas-wrap">
        <svg ref={svgRef} style={{ cursor: "move" }} viewBox={`${-margin} ${-margin} ${size} ${size}`}>
          <g>
            <path d={path(d3.geoGraticule10())} fill="none" stroke="#888" opacity="0.3" />
            {scene.links.map((link) => {
              const s = scene.nodes[link.source];
              const t = scene.nodes[link.target];
              if (!s || !t) return null;
              return (
                <path
                  key={`${link.source}:${link.target}`}
                  d={path({
                    type: "LineString",
                    coordinates: [
                      [s.lon, s.lat],
                      [t.lon, t.lat],
                    ],
                  })}
                  fill="none"
                  stroke="#999"
                  strokeWidth={link.strokeWidth}
                  opacity="0.55"
                />
              );
            })}
            {scene.nodes.map((node) => (
              <path
                key={node.id}
                d={path({ type: "Point", coordinates: [node.lon, node.lat] })}
                fill={node.color}
                stroke="#fff"
                strokeWidth="1.5"
              />
            ))}
          </g>
        </svg>
      </div>
    </div>
  );
}
