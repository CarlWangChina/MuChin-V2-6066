import { Navigate } from 'react-router-dom'

import { hasToken } from '../../../utils/auth'

function RequireAuth({ children }) {
  const authed = hasToken()

  return authed ? ( 
    children 
  ) : (
    <Navigate to="/login" replace /> 
  )
}

export default RequireAuth