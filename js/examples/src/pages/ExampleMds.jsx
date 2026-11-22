import { useEffect, useRef, useState } from "react";
import { classicalMds2d, normalizeLayout2d } from "egraph";
import { GraphCanvas } from "../components/GraphCanvas";
import { loadMiserablesGraph } from "../graphUtils";

export function ExampleMds() {
  const sceneRef = useRef(null);
  const [tick, setTick] = useState(0);
  const [status, setStatus] = useState("computing Classical MDS...");

  useEffect(() => {
    let cancelled = false;
    let graph = null;
    let layout = null;

    (async () => {
      try {
        const loaded = await loadMiserablesGraph();
        if (cancelled) {
          loaded.graph.free();
          return;
        }
        graph = loaded.graph;
        const t0 = performance.now();
        layout = classicalMds2d(graph);
        normalizeLayout2d(layout, 1.0, 42);
        const dt = (performance.now() - t0).toFixed(2);
        sceneRef.current = {
          coords: layout.coordsView(),
          numNodes: loaded.numNodes,
          nodes: loaded.nodes,
          links: loaded.links,
          nodeRadius: 5,
        };
        setStatus(`Classical MDS completed in ${dt} ms (${loaded.numNodes} nodes)`);
        setTick((t) => t + 1);
      } catch (err) {
        setStatus(`Error: ${err.message}`);
      }
    })();

    return () => {
      cancelled = true;
      layout?.free();
      graph?.free();
    };
  }, []);

  return (
    <div>
      <div className="graph-status-bar">
        <span className="tag is-info is-medium">Classical MDS (v6 classicalMds2d)</span>
        <span className="tag is-light">{status}</span>
      </div>
      <GraphCanvas sceneRef={sceneRef} tick={tick} />
    </div>
  );
}
