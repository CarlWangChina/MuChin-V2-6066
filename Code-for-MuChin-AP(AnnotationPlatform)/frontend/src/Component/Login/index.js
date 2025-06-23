import './index.css'

import { Button, Card, Form, Input } from 'antd'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { loginAsync } from '../../api/user'
import { setMusician, setToken, setUserDo, setUserName } from '../../utils/auth'
import withRouter from '../Base/WithRouter/withRouter'

const Login = () => {
  const initialState = {
    account: '',
    password: '',
  }

  const [loading, setLoading] = useState(false)

  const navigate = useNavigate()

  const onSubmit = (event) => {
    event.preventDefault()
  }

  const [user, setUser] = useState(initialState)

  const handleChange = (event) => {
    const { name, value } = event.target
    setUser({ ...user, [name]: value })
  }

  const onFinish = async () => {
    try {
      sessionStorage.clear()
      setLoading(true)
      const res = await loginAsync({ account: user.account.trim(), password: user.password })
      if (res.data.code === 200) {
        setToken(res.data.data.token)
        setUserName(user.account.trim())

        if (res.data.data.musician === 9) {
          navigate(`/check/${res.data.data.token}`)
          return
        }

        setUserDo(res.data.data.do)
        setMusician(res.data.data.musician)
        navigate('/home')
      } else {
        alert(res.data.msg)
        setLoading(false)
      }
    } catch (err) {
      if (String(err).includes('timeout')) {
        alert('请求超时！')
      } else if (String(err).includes('500')) {
        alert('请求失败！')
      } else {
        alert(err)
      }
      setLoading(false)
      return
    }
  }
  return (
    <div className="container">
      <Card title="登录" className="card">
        <Form
          name="login"
          initialValues={{ remember: true }}
          onSubmit={onSubmit}
          onFinish={onFinish}
          autoComplete="off"
        >
          <Form.Item
            label="账号"
            name="account"
            rules={[{ required: true, message: '请输入账号' }]}
          >
            <Input name="account" value={user.account || ''} onChange={handleChange} />
          </Form.Item>
          <Form.Item
            label="密码"
            name="password"
            rules={[{ required: true, message: '请输入密码' }]}
          >
            <Input.Password name="password" value={user.password || ''} onChange={handleChange} />
          </Form.Item>
          <Form.Item style={{ marginBottom: 0 }}>
            <Button type="primary" htmlType="submit" style={{ width: '100%' }} loading={loading}>
              登录
            </Button>
          </Form.Item>
        </Form>
      </Card>
    </div>
  )
}

export default withRouter(Login)