import './index.css'

import { Button, Space } from 'antd'
import Base64 from 'base-64'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import {
  fetchBzSumAsync,
  fetchMoreMusicsAsync,
  fetchMoreQcMusicsAsync,
  fetchMusicsAsync,
} from '../../api/musics'
import { getToken, getUserDo, isQC, setAlert, setFirst } from '../../utils/auth'

const Musics = () => {
  const [musics, setMusics] = useState([])
  const navigate = useNavigate()
  const [firstLoading, setFirstLoading] = useState(true)
  const [sum, setSum] = useState('')

  useEffect(() => {
    let ignore = false

    const token = getToken()

    async function startFetching() {
      try {
        const json = await fetchMusicsAsync(token)
        if (!ignore) {
          if (json.data.code === 200) {
            setMusics(json.data.data.song_list)
            setFirstLoading(false)
          } else {
            alert(json.data.msg)
          }
        }
      } catch (err) {
        if (String(err).includes('timeout')) {
          alert('请求超时！')
        } else if (String(err).includes('500')) {
          alert('请求失败！')
        } else {
          alert(err)
        }
      }
    }
    startFetching()
    return () => {
      ignore = true
    }
  }, [])

  useEffect(() => {
    if (isQC() === true) {
      return
    }

    let ignore = false
    async function startFetching() {
      try {
        const res = await fetchBzSumAsync(getToken())
        if (!ignore) {
          if (res.data.code === 200) {
            if (res.data.data.value.length > 0) {
              let sum_temp = ''
              res.data.data.value.forEach((item) => {
                sum_temp += `<div><span style="font-size: larger">${item.title}：</span><span style="color: red; font-weight: bold; font-size: larger">${item.count}</span></div>`
              })
              setSum(sum_temp)
            }
          } else {
            alert(res.data.msg)
          }
        }
      } catch (error) {
        if (String(error).includes('timeout')) {
          alert('请求超时！')
        } else if (String(error).includes('500')) {
          alert('请求失败！')
        } else {
          alert(error)
        }
      }
    }
    startFetching()
    return () => {
      ignore = true
    }
  }, [])

  const onClickC = (music_id, isMusic, work_id) => {
    if (isQC() === false) {
      const music = Base64.encode(`{"id": "${music_id}", "step": 1}`)
      const lyric = Base64.encode(`{"id": "${music_id}", "step": 2, "type": 0}`) 
      if (isMusic) {
        setFirst(true)
        setAlert(true)
        navigate(`/annotation/${music}`)
      } else {
        navigate(`/annotation/${lyric}`)
      }
    } else {

      const music = Base64.encode(`{"id": "${music_id}", "step": 1, "work_id": "${work_id}"}`)
      const lyric = Base64.encode(`{"id": "${music_id}", "step": 2, "work_id": "${work_id}"}`) 

      if (isMusic) {
        navigate(`/quality/${music}`)
      } else {
        navigate(`/quality/${lyric}`)
      }
    }
  }

  const onRefresh = async () => {
    if (musics.length > 0) {
      return
    }
    try {
      const res = await fetchMoreMusicsAsync(getToken())
      if (Number(res.data.code) === 200) {
        setMusics(res.data.data.song_list)
      } else {
        alert(res.data.msg)
      }
    } catch (error) {
      alert(error)
    }
  }

  const onRefreshQc = async () => {
    if (musics.length > 0) {
      return
    }
    try {
      const res = await fetchMoreQcMusicsAsync(getToken())
      if (Number(res.data.code) === 200) {
        setMusics(res.data.data.song_list)
      } else {
        alert(res.data.msg)
      }
    } catch (error) {
      alert(error)
    }
  }

  const qcListC = () => {
    return musics.length > 0 ? (
      musics.map((item) => {
        const ButtonComp = () => {
          if (item.editable === 3) {
            return (
              <div>
                <Button
                  type="primary"
                  size="small"
                  className="small_button"
                  onClick={() => onClickC(item.music_id, false, item.work_id)}
                >
                  词
                </Button>
                <Button
                  type="primary"
                  size="small"
                  className="small_button"
                  onClick={() => onClickC(item.music_id, true, item.work_id)}
                >
                  曲
                </Button>
              </div>
            )
          }
          if (item.editable === 2) {
            return (
              <Button
                type="primary"
                size="small"
                className="small_button"
                onClick={() => onClickC(item.music_id, false, item.work_id)}
              >
                词
              </Button>
            )
          }
          if (item.editable === 1) {
            return (
              <Button
                type="primary"
                size="small"
                className="small_button"
                onClick={() => onClickC(item.music_id, true, item.work_id)}
              >
                曲
              </Button>
            )
          }
        }

        return (
          <li className="music_item" key={item.work_id}>
            <h2 className="music_name_h2">{item.music_name}</h2>
            <ButtonComp />
          </li>
        )
      })
    ) : (
      <Button type="primary" className="refresh_btn" onClick={() => onRefreshQc()}>
        领取歌曲质检
      </Button>
    )
  }

  const fqcListC = () => {
    return musics.length > 0 ? (
      musics.map((item) => {
        const ButtonComp = () => {
          if (item.editable === 3) {
            return (
              <Space size="middle">
                <Button
                  type="primary"
                  size="small"
                  className="small_button"
                  onClick={() => onClickC(item.music_id, false)}
                >
                  词
                </Button>
                <Button
                  type="primary"
                  size="small"
                  className="small_button"
                  onClick={() => onClickC(item.music_id, true)}
                >
                  曲
                </Button>
              </Space>
            )
          }
          if (item.editable === 2) {
            return (
              <Space size="middle">
                <Button
                  type="primary"
                  size="small"
                  className="small_button"
                  onClick={() => onClickC(item.music_id, false)}
                >
                  词
                </Button>
              </Space>
            )
          }
          if (item.editable === 1) {
            return (
              <Space size="middle">
                <Button
                  type="primary"
                  size="small"
                  className="small_button"
                  onClick={() => onClickC(item.music_id, true)}
                >
                  曲
                </Button>
              </Space>
            )
          }
        }

        return (
          <li className="music_item" key={item.music_id}>
            <h2 className="music_name_h2">{item.music_name}</h2>
            <ButtonComp />
          </li>
        )
      })
    ) : getUserDo() === '1' ? (
      <Button type="primary" className="refresh_btn" onClick={() => onRefresh()}>
        领取歌曲标注
      </Button>
    ) : null
  }

  return (
    <div>
      {isQC() === false ? (
        <div className="total_sum">
          <div dangerouslySetInnerHTML={{ __html: sum }}></div>
        </div>
      ) : null}
      <ul style={{ padding: 0, margin: 0 }} className="music_list">
        {firstLoading === false ? (isQC() === true ? qcListC() : fqcListC()) : null}
      </ul>
    </div>
  )
}
export default Musics