import { createRoot } from "react-dom/client";
import { BrowserRouter as Router, Link, Route, Routes } from "react-router-dom";
import {
  ExampleEdgeBundling,
  ExampleForceAtlas2,
  ExampleGuMap,
  ExampleHyperbolicGeometry,
  ExampleKamadaKawai,
  ExampleMds,
  ExamplePrismOverlapRemoval,
  ExampleSgd,
  ExampleSphericalGeometry,
  ExampleStressMajorization,
  ExampleTFdp,
  ExampleTorus,
  ExampleTsNetTrackA,
  Home,
} from "./pages";

createRoot(document.getElementById("content")).render(
  <Router>
    <div>
      <header>
        <nav className="navbar is-primary">
          <div className="container">
            <div className="navbar-brand">
              <Link to="/" className="navbar-item">
                <strong>egraph v6 Examples</strong>
              </Link>
            </div>
          </div>
        </nav>
      </header>
      <section className="section">
        <div className="container">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/tsnet-track-a" element={<ExampleTsNetTrackA />} />
            <Route path="/t-fdp" element={<ExampleTFdp />} />
            <Route path="/force-atlas2" element={<ExampleForceAtlas2 />} />
            <Route path="/gumap" element={<ExampleGuMap />} />
            <Route path="/edge-bundling" element={<ExampleEdgeBundling />} />
            <Route path="/hyperbolic-geometry" element={<ExampleHyperbolicGeometry />} />
            <Route path="/kamada-kawai" element={<ExampleKamadaKawai />} />
            <Route path="/mds" element={<ExampleMds />} />
            <Route path="/prism-overlap-removal" element={<ExamplePrismOverlapRemoval />} />
            <Route path="/overwrap-removal" element={<ExamplePrismOverlapRemoval />} />
            <Route path="/sgd" element={<ExampleSgd />} />
            <Route path="/spherical-geometry" element={<ExampleSphericalGeometry />} />
            <Route path="/stress-majorization" element={<ExampleStressMajorization />} />
            <Route path="/torus-geometry" element={<ExampleTorus />} />
          </Routes>
        </div>
      </section>
    </div>
  </Router>,
);
