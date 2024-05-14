import { createRoot } from "react-dom/client";
import { BrowserRouter as Router, Route, Link } from "react-router-dom";
import {
  Home,
  ExampleEdgeBundling,
  ExampleKamadaKawai,
  ExampleHyperbolicGeometry,
  ExampleMds,
  ExampleSgd,
  ExampleSphericalGeometry,
  ExampleStressMajorization,
  ExampleTorus,
  ExampleOverwrapRemoval,
} from "./pages";

createRoot(document.getElementById("content")).render(
  <Router>
    <div>
      <header>
        <nav className="navbar is-primary">
          <div className="container">
            <div className="navbar-brand">
              <Link to="/" className="navbar-item">
                <h1>Egraph Examples</h1>
              </Link>
            </div>
          </div>
        </nav>
      </header>
      <section className="section">
        <div className="container">
          <Route path="/" component={Home} exact />
          <Route path="/edge-bundling" component={ExampleEdgeBundling} />
          <Route
            path="/hyperbolic-geometry"
            component={ExampleHyperbolicGeometry}
          />
          <Route path="/kamada-kawai" component={ExampleKamadaKawai} />
          <Route path="/mds" component={ExampleMds} />
          <Route path="/overwrap-removal" component={ExampleOverwrapRemoval} />
          <Route path="/sgd" component={ExampleSgd} />
          <Route
            path="/spherical-geometry"
            component={ExampleSphericalGeometry}
          />
          <Route
            path="/stress-majorization"
            component={ExampleStressMajorization}
          />
          <Route path="/torus-geometry" component={ExampleTorus} />
        </div>
      </section>
    </div>
  </Router>
);
