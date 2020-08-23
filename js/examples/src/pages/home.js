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
