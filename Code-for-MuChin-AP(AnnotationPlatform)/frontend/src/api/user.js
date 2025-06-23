import service from '../utils/axios'

export const loginAsync = (params) => service.post('user/login', params)
