import "eg-renderer";
import React from "react";
import { render } from "react-dom";
import { BrowserRouter as Router, Route, Link } from "react-router-dom";
import {
  Home,
  ExampleEdgeBundling,
  ExampleFm3,
  ExampleForceAtlas2,
  ExampleForceDirected,
  ExampleFruchtermanReingold,
  ExampleGroupInABox,
  ExampleKamadaKawai,
  ExampleHyperbolicGeometry,
  ExampleMds,
  ExampleSgd,
  ExampleSphericalGeometry,
  ExampleStressMajorization,
} from "./pages";

render(
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
          <Route path="/fm3" component={ExampleFm3} />
          <Route path="/force-atlas2" component={ExampleForceAtlas2} />
          <Route path="/force-directed" component={ExampleForceDirected} />
          <Route
            path="/fruchterman-reingold"
            component={ExampleFruchtermanReingold}
          />
          <Route path="/group-in-a-box" component={ExampleGroupInABox} />
          <Route
            path="/hyperbolic-geometry"
            component={ExampleHyperbolicGeometry}
          />
          <Route path="/kamada-kawai" component={ExampleKamadaKawai} />
          <Route path="/mds" component={ExampleMds} />
          <Route path="/sgd" component={ExampleSgd} />
          <Route
            path="/spherical-geometry"
            component={ExampleSphericalGeometry}
          />
          <Route
            path="/stress-majorization"
            component={ExampleStressMajorization}
          />
        </div>
      </section>
    </div>
  </Router>,
  document.getElementById("content")
);
