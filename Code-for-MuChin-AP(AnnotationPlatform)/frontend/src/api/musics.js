import service from '../utils/axios'

export const fetchMusicsAsync = (token) => service.get(`music/list/${token}`)

export const fetchMoreMusicsAsync = (token) => service.get(`music/refresh/new/${token}`)

export const fetchMoreQcMusicsAsync = (token) => service.get(`examiner/new/task/${token}`)

export const fetchBzSumAsync = (token) => service.get(`music/user/annotation/${token}`)