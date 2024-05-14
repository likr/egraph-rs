import React from "react";
import { Link } from "react-router-dom";

export class Home extends React.Component {
  render() {
    return (
      <div>
        <ul>
          <li>
            <Link to="/kamada-kawai">KamadaKawai</Link>
          </li>
          <li>
            <Link to="/stress-majorization">StressMajorization</Link>
          </li>
          <li>
            <Link to="/sgd">SGD</Link>
          </li>
          <li>
            <Link to="/mds">MDS</Link>
          </li>
          <li>
            <Link to="/hyperbolic-geometry">Hyperbolic Geometry</Link>
          </li>
          <li>
            <Link to="/spherical-geometry">Spherical Geometry</Link>
          </li>
          <li>
            <Link to="/torus-geometry">Torus Geometry</Link>
          </li>
          <li>
            <Link to="/edge-bundling">Edge-bundling</Link>
          </li>
          <li>
            <Link to="/overwrap-removal">Overwrap Removal</Link>
          </li>
        </ul>
      </div>
    );
  }
}
