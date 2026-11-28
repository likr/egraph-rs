import { useEffect, useRef, useState } from "react";
import { Layout2d, PrismOverlapRemoval, SgdSolver, runProgressive } from "egraph";
import { GraphCanvas } from "../components/GraphCanvas";
import { loadMiserablesGraph } from "../graphUtils";

export function ExamplePrismOverlapRemoval() {
  const sceneRef = useRef(null);
  const [tick, setTick] = useState(0);
  const [status, setStatus] = useState("initializing...");

  useEffect(() => {
    const ac = new AbortController();
    let graph = null;
    let layout = null;
    let sgd = null;
    let prism = null;

    (async () => {
      try {
        const loaded = await loadMiserablesGraph();
        if (ac.signal.aborted) {
          loaded.graph.free();
          return;
        }
        graph = loaded.graph;
        layout = Layout2d.init(loaded.numNodes, { seed: 4 });
        sgd = SgdSolver.full(graph, { iterations: 60, eps: 0.1, seed: 4 });

        sceneRef.current = {
          coords: layout.coordsView(),
          numNodes: loaded.numNodes,
          nodes: loaded.nodes,
          links: loaded.links,
          nodeRadius: 11,
        };
        setTick((t) => t + 1);

        await runProgressive(
          sgd,
          layout,
          (ctx) => {
            sceneRef.current.coords = layout.coordsView();
            setStatus(`Phase 1 (SGD): ${ctx.iteration}/${ctx.totalIterations}`);
            setTick((t) => t + 1);
          },
          { frameBudgetMs: 10, signal: ac.signal },
        );

        const radii = new Float32Array(loaded.numNodes).fill(0.45);
        prism = PrismOverlapRemoval.fromRadii(radii, { maxIterations: 80, padding: 0.05 });

        await runProgressive(
          prism,
          layout,
          (ctx) => {
            sceneRef.current.coords = layout.coordsView();
            setStatus(
              `Phase 2 (PRISM): step ${ctx.iteration}/${ctx.totalIterations}, overlaps=${prism.lastOverlapCount} (${ctx.status})`,
            );
            setTick((t) => t + 1);
          },
          { frameBudgetMs: 10, signal: ac.signal },
        );
      } catch (err) {
        if (err?.name !== "AbortError") {
          setStatus(`Error: ${err.message}`);
        }
      }
    })();

    return () => {
      ac.abort();
      prism?.free();
      sgd?.free();
      layout?.free();
      graph?.free();
    };
  }, []);

  return (
    <div>
      <div className="graph-status-bar">
        <span className="tag is-info is-medium">PRISM Overlap Removal (PrismOverlapRemoval + runProgressive)</span>
        <span className="tag is-light">{status}</span>
      </div>
      <GraphCanvas sceneRef={sceneRef} tick={tick} />
    </div>
  );
}
