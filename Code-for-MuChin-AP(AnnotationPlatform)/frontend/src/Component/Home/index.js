import './index.css'

import { Component } from 'react'

import withRouter from '../Base/WithRouter/withRouter'
import Header from '../Header'
import Musics from '../Musics'

class Home extends Component {
  render() {
    return (
      <div className="home">
        <Header />
        <Musics />
      </div>
    )
  }
}

export default withRouter(Home)