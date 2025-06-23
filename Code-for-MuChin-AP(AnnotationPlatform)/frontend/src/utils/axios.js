import axios from 'axios'

axios.defaults.headers['Content-Type'] = 'application/json;charset=utf-8'

const service = axios.create({
  baseURL:
    process.env.NODE_ENV === 'development' ? 'http://0.0.0.0:32000' : 'http://52.53.193.231:8080',
  timeout: 1000 * 300,
})

export default service