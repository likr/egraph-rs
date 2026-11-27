import { useEffect, useRef, useState } from "react";
import { FdebSolver, Layout2d, SgdSolver, runProgressive } from "egraph";
import { GraphCanvas } from "../components/GraphCanvas";
import { loadMiserablesGraph } from "../graphUtils";

export function ExampleEdgeBundling() {
  const sceneRef = useRef(null);
  const [tick, setTick] = useState(0);
  const [status, setStatus] = useState("initializing...");

  useEffect(() => {
    const ac = new AbortController();
    let graph = null;
    let layout = null;
    let sgd = null;
    let fdeb = null;

    (async () => {
      try {
        const loaded = await loadMiserablesGraph();
        if (ac.signal.aborted) {
          loaded.graph.free();
          return;
        }
        graph = loaded.graph;
        layout = Layout2d.init(loaded.numNodes, 4);
        sgd = SgdSolver.full(graph, { iterations: 80, eps: 0.1, seed: 4 });
        sgd.run(layout);

        sceneRef.current = {
          coords: layout.coordsView(),
          numNodes: loaded.numNodes,
          nodes: loaded.nodes,
          links: loaded.links,
          nodeRadius: 4.5,
        };
        setTick((t) => t + 1);

        fdeb = new FdebSolver(graph, { cycles: 4, stepsPerCycle: 15, s0: 0.08 });

        await runProgressive(
          fdeb,
          layout,
          (ctx) => {
            sceneRef.current.coords = layout.coordsView();
            sceneRef.current.bundled = {
              offsets: fdeb.edgeOffsets(),
              points: fdeb.points(),
            };
            setStatus(`FDEB: step ${ctx.iteration}/${ctx.totalIterations} (${ctx.status})`);
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
      fdeb?.free();
      sgd?.free();
      layout?.free();
      graph?.free();
    };
  }, []);

  return (
    <div>
      <div className="graph-status-bar">
        <span className="tag is-info is-medium">Force-Directed Edge Bundling (FdebSolver + runProgressive)</span>
        <span className="tag is-light">{status}</span>
      </div>
      <GraphCanvas sceneRef={sceneRef} tick={tick} />
    </div>
  );
}
