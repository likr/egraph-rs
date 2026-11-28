import { Link } from "react-router-dom";

const demos = [
  { to: "/tsnet-track-a", title: "Track A: TsNetSolver.pure2Hop (fitFft + Adaptive k-NN)", isNew: true },
  { to: "/t-fdp", title: "t-FDP: TFdpSolver (ibFft + PivotMds)", isNew: true },
  { to: "/force-atlas2", title: "ForceAtlas2 (barnesHut + runProgressive)", isNew: true },
  { to: "/gumap", title: "GuMap (GuMapSolver.fromGraphDog + runProgressive)", isNew: true },
  { to: "/sgd", title: "Full SGD (SgdSolver.full + runProgressive)" },
  { to: "/kamada-kawai", title: "Kamada-Kawai (KamadaKawaiSolver.fromGraph + runProgressive)" },
  { to: "/stress-majorization", title: "Stress Majorization (StressMajorizationSolver + runProgressive)" },
  { to: "/mds", title: "Classical MDS (ClassicalMds + normalizeLayout2d)" },
  { to: "/edge-bundling", title: "Force-Directed Edge Bundling (EdgePolylines + FdebSolver + runProgressive)" },
  { to: "/prism-overlap-removal", title: "PRISM Overlap Removal (PrismOverlapRemoval + runProgressive)" },
  { to: "/hyperbolic-geometry", title: "Hyperbolic Geometry (ManifoldLayout2d + Poincaré Disk)" },
  { to: "/spherical-geometry", title: "Spherical Geometry (ManifoldLayout2d + Azimuthal Globe)" },
  { to: "/torus-geometry", title: "Torus 2D Geometry (ManifoldLayout2d + Tiled Torus)" },
];

export function Home() {
  return (
    <div className="content">
      <h2 className="title is-4">egraph v6 WebAssembly Layout Demos</h2>
      <p className="subtitle is-6">
        Single-threaded Rust + WebAssembly layouts using zero-copy <code>Layout2d.coordsView()</code> and{" "}
        <code>runProgressive()</code> frame-budgeted scheduling.
      </p>
      <ul>
        {demos.map((demo) => (
          <li key={demo.to}>
            <Link to={demo.to}>{demo.title}</Link>{" "}
            {demo.isNew && <span className="tag is-primary is-light is-small">v6</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}
