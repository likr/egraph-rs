import { useEffect, useRef, useState } from "react";
import { ForceAtlas2Solver, PivotMds, runProgressive } from "egraph";
import { GraphCanvas } from "../components/GraphCanvas";
import { loadEmbeddedGraph } from "../graphUtils";

export function ExampleForceAtlas2() {
  const sceneRef = useRef(null);
  const [dataset, setDataset] = useState("lesmis");
  const [tick, setTick] = useState(0);
  const [status, setStatus] = useState("initializing...");

  useEffect(() => {
    const ac = new AbortController();
    let graph = null;
    let layout = null;
    let solver = null;

    (async () => {
      try {
        const loaded = await loadEmbeddedGraph(dataset);
        if (ac.signal.aborted) {
          loaded.graph.free();
          return;
        }
        graph = loaded.graph;
        const mds = new PivotMds({ numPivots: 32, seed: 42, targetStd: 1.0 });
        layout = mds.generate(graph);
        mds.free();
        solver = new ForceAtlas2Solver(graph, {
          iterations: 200,
          backend: "barnesHut",
          linlogMode: false,
          seed: 42,
        });
        sceneRef.current = {
          coords: layout.coordsView(),
          numNodes: loaded.numNodes,
          nodes: loaded.nodes,
          links: loaded.links,
          nodeRadius: loaded.numNodes > 200 ? 2.5 : 5,
        };
        setTick((t) => t + 1);

        const t0 = performance.now();
        await runProgressive(
          solver,
          layout,
          (ctx) => {
            sceneRef.current.coords = layout.coordsView();
            setStatus(
              `ForceAtlas2 (${dataset}): step ${ctx.iteration}/${ctx.totalIterations} (${ctx.status})`,
            );
            setTick((t) => t + 1);
          },
          { frameBudgetMs: 10, signal: ac.signal },
        );
        const dt = (performance.now() - t0).toFixed(1);
        setStatus(`ForceAtlas2 completed in ${dt} ms (${loaded.numNodes} nodes)`);
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
  }, [dataset]);

  return (
    <div>
      <div className="graph-status-bar">
        <span className="tag is-primary is-medium">ForceAtlas2 (barnesHut + runProgressive)</span>
        <div className="select is-small">
          <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
            <option value="lesmis">lesmis (77 nodes)</option>
            <option value="karate">karate (34 nodes)</option>
            <option value="1138_bus">1138_bus (1,138 nodes)</option>
          </select>
        </div>
        <span className="tag is-light">{status}</span>
      </div>
      <GraphCanvas sceneRef={sceneRef} tick={tick} />
    </div>
  );
}
