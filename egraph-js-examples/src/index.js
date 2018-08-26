import '@webcomponents/custom-elements'
import 'eg-renderer'
import React from 'react'
import {render} from 'react-dom'
import {BrowserRouter as Router, Route} from 'react-router-dom'
import {Home} from './pages/home'
import {ExampleForceDirected} from './pages/example-force-directed'
import {ExampleGroupInABox} from './pages/example-group-in-a-box'

render(<Router>
  <div>
    <Route path='/' component={Home} exact />
    <Route path='/force-directed' component={ExampleForceDirected} />
    <Route path='/group-in-a-box' component={ExampleGroupInABox} />
  </div>
</Router>, document.getElementById('content'))
