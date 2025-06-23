import Base64 from 'base-64'

const TOKEN_KEY = 'a_u_t_k'

const USER_NAME_KEY = 'u_n_k'

const FIRST_KEY = 'f_m_k'

const ALERT_KEY = 'a_t_m_k'

const Musician_KEY = 'musician_t_k'

const ZJ_CHECK_TYPE_KEY = 'z_c_t_k'

const BZ_MUSICS_DO_KEY = 'u_m_d_k'

export function getToken() {
  const encode_token = sessionStorage.getItem(TOKEN_KEY)
  const decode_token = Base64.decode(encode_token)
  return decode_token
}

export function setToken(token) {
  const encode_token = Base64.encode(token)
  sessionStorage.setItem(TOKEN_KEY, encode_token)
}

export function removeToken() {
  sessionStorage.removeItem(TOKEN_KEY)
}

export function hasToken() {
  const tk = sessionStorage.getItem(TOKEN_KEY)
  if (tk === null || tk === undefined) {
    return false
  } else {
    return true
  }
}

export function getUserName() {
  const encode_name = sessionStorage.getItem(USER_NAME_KEY)
  const decode_name = Base64.decode(encode_name)
  return decode_name
}

export function setUserName(name) {
  const encode_name = Base64.encode(name)
  sessionStorage.setItem(USER_NAME_KEY, encode_name)
}

export function removeUserName() {
  sessionStorage.removeItem(USER_NAME_KEY)
}

export function setFirst(first) {
  const encode_first = Base64.encode(first)
  sessionStorage.setItem(FIRST_KEY, encode_first)
}

export function getFirst() {
  const encode_first = sessionStorage.getItem(FIRST_KEY)
  const decode_first = Base64.decode(encode_first)
  return decode_first
}

export function removeFirst() {
  sessionStorage.removeItem(FIRST_KEY)
}

export function setAlert(alert) {
  const encode_alert = Base64.encode(alert)
  sessionStorage.setItem(ALERT_KEY, encode_alert)
}

export function getAlert() {
  const encode_alert = sessionStorage.getItem(ALERT_KEY)
  const decode_alert = Base64.decode(encode_alert)
  return decode_alert
}

export function setMusician(musician) {
  const encode_musician = Base64.encode(musician)
  sessionStorage.setItem(Musician_KEY, encode_musician)
}

export function getMusician() {
  const encode_musician = sessionStorage.getItem(Musician_KEY)
  const decode_musician = Base64.decode(encode_musician)
  const musician = parseInt(decode_musician)
  return musician
}

export function isQC() {
  const encode_musician = sessionStorage.getItem(Musician_KEY)
  const decode_musician = Base64.decode(encode_musician)
  const musician = parseInt(decode_musician)
  return musician === 3 || musician === 30 || musician === 4
}

export function setZjType(type) {
  const encode_type = Base64.encode(type)
  sessionStorage.setItem(ZJ_CHECK_TYPE_KEY, encode_type)
}

export function getZjType() {
  const encode_type = sessionStorage.getItem(ZJ_CHECK_TYPE_KEY)
  const decode_type = Base64.decode(encode_type)
  return decode_type
}

export function hasZjType() {
  const type = sessionStorage.getItem(ZJ_CHECK_TYPE_KEY)
  if (type === null || type === undefined) {
    return false
  } else {
    return true
  }
}

export function removeZjType() {
  sessionStorage.removeItem(ZJ_CHECK_TYPE_KEY)
}

export function getUserDo() {
  const encode_do = sessionStorage.getItem(BZ_MUSICS_DO_KEY)
  const decode_do = Base64.decode(encode_do)
  return decode_do
}

export function setUserDo(userDo) {
  const encode_do = Base64.encode(userDo)
  sessionStorage.setItem(BZ_MUSICS_DO_KEY, encode_do)
}