import '@webcomponents/custom-elements'
import 'eg-renderer'
import React from 'react'
import {render} from 'react-dom'
import {load} from 'egraph'
import {BrowserRouter as Router, Route, Link} from 'react-router-dom'
import {Home} from './pages/home'
import {ExampleForceDirected} from './pages/example-force-directed'
import {ExampleFM3} from './pages/example-fm3'
import {ExampleGroupInABox} from './pages/example-group-in-a-box'
import {ExampleEdgeConcentration} from './pages/example-edge-concentration'

load('/egraph.wasm').then(() => {
  render(<Router>
    <div>
      <header>
        <nav className='navbar is-primary'>
          <div className='container'>
            <div className='navbar-brand'>
              <Link to='/' className='navbar-item'>
                <h1>Egraph Examples</h1>
              </Link>
            </div>
          </div>
        </nav>
      </header>
      <section className='section'>
        <div className='container'>
          <Route path='/' component={Home} exact />
          <Route path='/force-directed' component={ExampleForceDirected} />
          <Route path='/fm3' component={ExampleFM3} />
          <Route path='/group-in-a-box' component={ExampleGroupInABox} />
          <Route path='/edge-concentration' component={ExampleEdgeConcentration} />
        </div>
      </section>
    </div>
  </Router>, document.getElementById('content'))
})
