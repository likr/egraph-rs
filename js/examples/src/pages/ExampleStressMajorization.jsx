import { useEffect, useRef, useState } from "react";
import { Layout2d, StressMajorizationSolver, runProgressive } from "egraph";
import { GraphCanvas } from "../components/GraphCanvas";
import { loadMiserablesGraph } from "../graphUtils";

export function ExampleStressMajorization() {
  const sceneRef = useRef(null);
  const [tick, setTick] = useState(0);
  const [status, setStatus] = useState("initializing...");

  useEffect(() => {
    const ac = new AbortController();
    let graph = null;
    let layout = null;
    let solver = null;

    (async () => {
      try {
        const loaded = await loadMiserablesGraph();
        if (ac.signal.aborted) {
          loaded.graph.free();
          return;
        }
        graph = loaded.graph;
        layout = Layout2d.init(loaded.numNodes, 7);
        solver = StressMajorizationSolver.fromGraph(graph, {
          epsilon: 1e-5,
          maxIterations: 100,
        });
        sceneRef.current = {
          coords: layout.coordsView(),
          numNodes: loaded.numNodes,
          nodes: loaded.nodes,
          links: loaded.links,
          nodeRadius: 5,
        };
        setTick((t) => t + 1);

        await runProgressive(
          solver,
          layout,
          (ctx) => {
            sceneRef.current.coords = layout.coordsView();
            const stress = solver.lastStress();
            const stressStr = typeof stress === "number" ? `, stress=${stress.toFixed(2)}` : "";
            setStatus(`Stress Majorization: step ${ctx.iteration}/${ctx.totalIterations} (${ctx.status}${stressStr})`);
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
      solver?.free();
      layout?.free();
      graph?.free();
    };
  }, []);

  return (
    <div>
      <div className="graph-status-bar">
        <span className="tag is-info is-medium">Stress Majorization (v6 CsrGraph + runProgressive)</span>
        <span className="tag is-light">{status}</span>
      </div>
      <GraphCanvas sceneRef={sceneRef} tick={tick} />
    </div>
  );
}
