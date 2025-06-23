import service from '../utils/axios'

export const fetchLyricsAsync = ({ song_id, token }, type) =>
  service.get(`music/song_lyc/${song_id}/${type}/${token}`)
export const fetchLyricsAsync2 = ({ song_id, token }, type) =>
  service.get(`music/detail/${song_id}/${type}/${token}`)

export const pushMusicInfoAsync = ({ song_id, token }, params) =>
  service.post(`music/detail/save/${song_id}/${token}`, params)

export const pushLyricsAsync = ({ song_id, token }, params) =>
  service.post(`music/song_lyc/${song_id}/${token}`, params)

export const fetchSectionLyricsAsync = ({ song_id, token }) =>
  service.get(`music/song_lyc/section/${song_id}/${token}`)

export const pushSectionLyricsAsync = ({ song_id, token }, params) =>
  service.post(`music/song_lyc/section/${song_id}/${token}`, params)

export const fetchRhymeLyricsAsync = ({ song_id, token }) =>
  service.get(`music/song_lyc/rhyme/${song_id}/${token}`)

export const pushRhymeLyricsAsync = ({ song_id, token }, params) =>
  service.post(`music/song_lyc/rhyme/${song_id}/${token}`, params)

export const pushRhymeEnterAsync = (token, params) =>
  service.post(`music/song_lyc/rhyme/calcul/${token}`, params)

export const fetchMusicQusAsync = ({ song_id, token }) =>
  service.get(`investigate/detail/${song_id}/${token}`)

export const fetchMusicTagsAsync = (tag_id, token) =>
  service.get(`investigate/category/${tag_id}/${token}`)

export const pushMusicAnsAsync = (song_id, token, params) =>
  service.post(`investigate/detail/${song_id}/${token}`, params)

export const pushLyricLansAsync = ({ song_id, token }, params, type) =>
  service.post(`music/song/lan/check/${type}/${song_id}/${token}`, params)

export const pushPendingReasonAsync = (song_id, token, params) =>
  service.post(`music/pending/${song_id}/${token}`, params)

export const pushMusicPendingReasonAsync = (song_id, token, params) =>
  service.post(`investigate/pending/${song_id}/${token}`, params)