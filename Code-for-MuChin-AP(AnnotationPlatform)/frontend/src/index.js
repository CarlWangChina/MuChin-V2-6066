import './index.css'
import 'lib-flexible'

import ReactDOM from 'react-dom/client'
import { HashRouter } from 'react-router-dom'

import App from './App'
import reportWebVitals from './reportWebVitals'

const root = ReactDOM.createRoot(document.getElementById('root'))

root.render(
  <HashRouter>
    <App />
  </HashRouter>
)

reportWebVitals()