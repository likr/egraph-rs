import "eg-renderer";
import React from "react";
import { render } from "react-dom";
import { BrowserRouter as Router, Route, Link } from "react-router-dom";
import {
  Home,
  ExampleEdgeBundling,
  ExampleFm3,
  ExampleForceDirected,
  ExampleGroupInABox,
  ExampleKamadaKawai,
  ExampleNonEuclideanForceSimulation,
  ExampleStressMajorization,
} from "./pages";
import { ExampleFruchtermanReingold } from "./pages/ExampleFruchtermanReingold";

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
          <Route path="/force-directed" component={ExampleForceDirected} />
          <Route
            path="/fruchterman-reingold"
            component={ExampleFruchtermanReingold}
          />
          <Route path="/group-in-a-box" component={ExampleGroupInABox} />
          <Route path="/kamada-kawai" component={ExampleKamadaKawai} />
          <Route
            path="/non-euclidean-force-simulation"
            component={ExampleNonEuclideanForceSimulation}
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
