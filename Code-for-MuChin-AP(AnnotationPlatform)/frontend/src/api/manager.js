import service from '../utils/axios'

export const fetchManagerListAsync = (account, token) =>
  service.get(`manager/search/${account}/${token}`)

export const fetchManagerLyricDescAsync = (id, uid, token) =>
  service.get(`manager/detail/${uid}/${id}/${token}`)