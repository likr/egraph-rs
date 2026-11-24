import { useEffect, useRef, useState } from "react";
import { GuMapSolver, evaluateMetrics, pivotMdsInit, runProgressive } from "egraph";
import { GraphCanvas } from "../components/GraphCanvas";
import { loadEmbeddedGraph } from "../graphUtils";

export function ExampleGuMap() {
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
        layout = pivotMdsInit(graph, 32, 42, null, 1.0);
        solver = GuMapSolver.fromGraphDog(graph, {
          nEpochs: 150,
          negativeSampleRate: 5,
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
              `GuMap (${dataset}): epoch ${ctx.iteration}/${ctx.totalIterations} (${ctx.status})`,
            );
            setTick((t) => t + 1);
          },
          { frameBudgetMs: 10, signal: ac.signal },
        );
        const dt = (performance.now() - t0).toFixed(1);
        const report = evaluateMetrics(graph, layout, ["np2", "sns"], 16, 128, 42);
        const np2 = report.neighborhoodPreservation2Hop?.toFixed(4) ?? "n/a";
        const sns = report.scaleNormalizedStress?.toFixed(4) ?? "n/a";
        report.free();
        setStatus(`GuMap completed in ${dt} ms | NP2=${np2}, SNS=${sns}`);
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
        <span className="tag is-primary is-medium">GuMap (GuMapSolver.fromGraphDog + runProgressive)</span>
        <div className="select is-small">
          <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
            <option value="lesmis">lesmis (77 nodes)</option>
            <option value="1138_bus">1138_bus (1,138 nodes)</option>
            <option value="3_elt">3_elt (4,720 nodes)</option>
          </select>
        </div>
        <span className="tag is-light">{status}</span>
      </div>
      <GraphCanvas sceneRef={sceneRef} tick={tick} />
    </div>
  );
}
