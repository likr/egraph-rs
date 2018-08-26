import React from 'react'

export class Wrapper extends React.Component {
  constructor () {
    super()
    this.handler = () => {
      this.resize()
    }
  }

  componentDidMount () {
    this.resize()
    window.addEventListener('resize', this.handler)
  }

  componentWillUnmount () {
    window.removeEventListener('resize', this.handler)
  }

  render () {
    return <figure ref='wrapper' className='image is-3by2' style={{boxShadow: '0 0 1em', margin: '1rem'}}>
      {this.props.children}
    </figure>
  }

  resize () {
    const {onResize} = this.props
    if (onResize) {
      const {clientWidth, clientHeight} = this.refs.wrapper
      onResize(clientWidth, clientHeight)
    }
  }
}
