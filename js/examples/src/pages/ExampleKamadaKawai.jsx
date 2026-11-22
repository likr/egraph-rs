import { useEffect, useRef, useState } from "react";
import { KamadaKawaiSolver, Layout2d, runProgressive } from "egraph";
import { GraphCanvas } from "../components/GraphCanvas";
import { loadMiserablesGraph } from "../graphUtils";

export function ExampleKamadaKawai() {
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
        layout = Layout2d.init(loaded.numNodes, 42);
        solver = KamadaKawaiSolver.fromGraph(graph, {
          eps: 1e-3,
          maxInnerIters: 50,
          maxOuterIters: 400,
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
            setStatus(`Kamada-Kawai: iteration ${ctx.iteration}/${ctx.totalIterations} (${ctx.status})`);
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
        <span className="tag is-info is-medium">Kamada-Kawai (v6 CsrGraph + runProgressive)</span>
        <span className="tag is-light">{status}</span>
      </div>
      <GraphCanvas sceneRef={sceneRef} tick={tick} />
    </div>
  );
}
