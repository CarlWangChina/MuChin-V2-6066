import './App.css'

import { Component } from 'react'
import { Route, Routes } from 'react-router-dom'

import Annotation from './Component/Annotation'
import RequireAuth from './Component/Base/Auth'
import withRouter from './Component/Base/WithRouter/withRouter'
import Check from './Component/Check'
import CheckMusicShow from './Component/Check/MusicShow'
import CheckLyricShow from './Component/Check/Show'
import Home from './Component/Home'
import Login from './Component/Login'
import Quality from './Component/Quality'

class App extends Component {
  render() {
    return (
      <div className="App">
        <Routes>
          <Route path="/" element={<Login />} />
          <Route path="/login" element={<Login />} />
          <Route
            path="/home"
            element={
              <RequireAuth>
                <Home></Home>
              </RequireAuth>
            }
          />
          <Route
            path="/annotation/:is"
            element={
              <RequireAuth>
                <Annotation></Annotation>
              </RequireAuth>
            }
          />
          <Route
            path="/quality/:is"
            element={
              <RequireAuth>
                <Quality></Quality>
              </RequireAuth>
            }
          />
          <Route
            path="/check/:token"
            element={
              <RequireAuth>
                <Check></Check>
              </RequireAuth>
            }
          />
          <Route
            path="/check/lyric/:token/:userId/:work_id"
            element={
              <RequireAuth>
                <CheckLyricShow></CheckLyricShow>
              </RequireAuth>
            }
          />
          <Route
            path="/check/music/:token/:userId/:work_id"
            element={
              <RequireAuth>
                <CheckMusicShow></CheckMusicShow>
              </RequireAuth>
            }
          />
        </Routes>
      </div>
    )
  }
}

export default withRouter(App)