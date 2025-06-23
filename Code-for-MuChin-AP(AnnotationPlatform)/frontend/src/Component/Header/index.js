import './index.css'

import { Button } from 'antd'
import { useNavigate } from 'react-router-dom'

import { getMusician, getUserName, isQC } from '../../utils/auth'

const Header = () => {
  const navigate = useNavigate()

  const onClick = () => {
    sessionStorage.clear()
    navigate('/login')
  }

  return (
    <div className="head-row">
      <h2 className="user_name">{getUserName()}</h2>
      {isQC() === true ? (
        <h2 className="user_name" style={{ flex: 1 }}>
          {getMusician() === 4 ? '特殊质检员' : '质检员'}
        </h2>
      ) : (
        <h2 className="user_name" style={{ flex: 1 }}>
          {' '}
          {getMusician() === 2 ? '特殊标注员' : '标注员'}{' '}
        </h2>
      )}
      <Button className="logout_btn" onClick={onClick} type="primary" size="small">
        退出登录
      </Button>
    </div>
  )
}

export default Header