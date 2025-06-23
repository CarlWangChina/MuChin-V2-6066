import '../Check/index.scss'

import { Button, Input } from 'antd'
import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'

import { fetchManagerListAsync } from '../../api/manager'
import { getToken } from '../../utils/auth'
import withRouter from '../Base/WithRouter/withRouter'
import List from './List'

const Check = (props) => {
  const { token } = props.params

  const location = useLocation()
  const queryParams = new URLSearchParams(location.search)
  const searchValueUrl = queryParams.get('searchValue')

  const [searchValue, setSearchValue] = useState(searchValueUrl !== undefined ? searchValueUrl : '')

  const [musics, setMusics] = useState([])
  const [userId, setUserId] = useState('')

  const navigate = useNavigate()

  useEffect(() => {
    if (token !== getToken()) {
      alert('Token出错，请重新登录！')
      navigate('/login')
      return
    }
    if (searchValue.length > 0) {
      onSearch()
    }
  }, [token, navigate])

  const onSearchValueChange = (e) => {
    setSearchValue(e.target.value)
  }

  const onSearch = async () => {
    if (searchValue === '') {
      alert('请输入ID（账号）')
    } else {
      try {
        const res = await fetchManagerListAsync(searchValue.trim(), token)

        if (Number(res.data.code) === 200) {
          setUserId(res.data.data.uid)
          setMusics(res.data.data.song_list)
        } else {
          alert(res.data.msg)
        }
      } catch (error) {
        alert(error)
      }
    }
  }

  const onClickItem = (type, work_id) => {
    if (type === 1) {
      navigate(`/check/lyric/${token}/${userId}/${work_id}?searchValue=${searchValue}`)
    } else {
      navigate(`/check/music/${token}/${userId}/${work_id}?searchValue=${searchValue}`)
    }
  }

  return (
    <div className="check_page">
      <div style={{ display: 'flex', alignItems: 'center' }} className="check_search">
        <Input
          className="check_search_input"
          id="check_search_input"
          size="large"
          placeholder="请输入ID（账号）"
          value={searchValue}
          onChange={onSearchValueChange}
        />
        <Button type="primary" size="large" onClick={onSearch}>
          搜索
        </Button>
      </div>
      {<List musics={musics} onClickItem={onClickItem} />}
    </div>
  )
}

export default withRouter(Check)