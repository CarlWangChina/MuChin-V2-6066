import service from '../utils/axios'

export const fetchCheckLyricsAsync = ({ song_id, token }) =>
  service.get(`examiner/check/detail/${song_id}/${token}`)

export const pushCheckSelectedAsync = (song_id, token, data) =>
  service.post(`examiner/check/${song_id}/${token}`, data)

export const pushCheckSubmitAsync = ({ song_id, token }, data, type) =>
  service.post(`examiner/detail/save/${type}/${song_id}/${token}`, data)
export const pushCheckSubmitAsync2 = ({ song_id, token }, data, type) =>
  service.post(`examiner/detail/save2/${type}/${song_id}/${token}`, data)

export const fetchCheckMusicAsync = ({ song_id, token }) =>
  service.get(`examiner/check/investigate/detail/${song_id}/${token}`)
export const fetchCheckMusicAsyncSpecail = ({ uid, work_id, token }) =>
  service.get(`manager/detail/qa/${uid}/${work_id}/${token}`)

export const pushCheckMusicScoreAsync = (song_id, token, data) =>
  service.post(`examiner/check/investigate/grade/${song_id}/${token}`, data)

export const pushCheckPendingReasonAsync = (song_id, token, data) =>
  service.post(`examiner/check/pending/${song_id}/${token}`, data)