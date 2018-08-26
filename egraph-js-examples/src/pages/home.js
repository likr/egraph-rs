import React from 'react'
import {Link} from 'react-router-dom'

export class Home extends React.Component {
  render () {
    return <div>
      <ul>
        <li><Link to='/force-directed'>force-directed</Link></li>
        <li><Link to='/group-in-a-box'>group-in-a-box</Link></li>
      </ul>
    </div>
  }
}
