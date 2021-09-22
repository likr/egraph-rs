import React from "react";
import { Link } from "react-router-dom";

export class Home extends React.Component {
  render() {
    return (
      <div>
        <ul>
          <li>
            <Link to="/force-directed">Force-directed Layout</Link>
          </li>
          <li>
            <Link to="/fruchterman-reingold">FruchtermanReingold</Link>
          </li>
          <li>
            <Link to="/kamada-kawai">KamadaKawai</Link>
          </li>
          <li>
            <Link to="/stress-majorization">StressMajorization</Link>
          </li>
          <li>
            <Link to="/non-euclidean-force-simulation">
              NonEuclideanForceSimulation
            </Link>
          </li>
          <li>
            <Link to="/spherical-embedding">SphericalEmbedding</Link>
          </li>
          <li>
            <Link to="/fm3">FM3</Link>
          </li>
          <li>
            <Link to="/group-in-a-box">Group-in-a-box Layout</Link>
          </li>
          <li>
            <Link to="/edge-bundling">Edge-bundling</Link>
          </li>
          {/*<li>
            <Link to="/dag">dag</Link>
          </li>
          <li>
            <Link to="/biclustering">biclustering</Link>
          </li>*/}
        </ul>
      </div>
    );
  }
}
