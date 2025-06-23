import { Button } from 'antd'
import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'

import { fetchManagerLyricDescAsync } from '../../../api/manager'
import initColors from '../../../utils/colors'
import AudioItem from '../../Base/Audio'
import withRouter from '../../Base/WithRouter/withRouter'

const CheckLyricShow = (props) => {
  const { work_id, userId, token } = props.params

  const location = useLocation()
  const queryParams = new URLSearchParams(location.search)
  const searchValue = queryParams.get('searchValue')
  const navigate = useNavigate()
  const [songName, setSongName] = useState('')
  const [songUrl, setSongUrl] = useState('')
  const [lyrics, setLyrics] = useState([])
  const [reasons, setReasons] = useState([])
  const [pending, setPending] = useState('')

  useEffect(() => {
    let ignore = false

    async function startFetching() {
      try {
        const res = await fetchManagerLyricDescAsync(work_id, userId, token)
        if (!ignore) {
          if (Number(res.data.code) === 200) {
            const data = res.data.data

            setSongName(data.song_name)
            setSongUrl(data.song_url)

            if (Object.prototype.hasOwnProperty.call(data, 'lyric')) {
              let currentType
              const map = new Map()
              let i = 0

              const l = []

              data.lyric.forEach((item, index) => {
                const item_type = item.type

                if (index === 0) {
                  currentType = item_type
                }
                if (item_type !== currentType) {
                  currentType = item_type
                  map.clear()
                }

                let co = ''
                let item_rh
                if (Object.prototype.hasOwnProperty.call(item, 'rhyme') || item.rhyme === '') {
                  item_rh = item.rhyme
                  if (!map.has(item_rh)) {
                    const ic = initColors[i]
                    map.set(item_rh, ic)
                    i++
                  }
                  co = map.get(item_rh)
                } else {
                  item_rh = ''
                  co = ''
                }

                const n_item = {
                  type: currentType,
                  words: item.words,
                  rhyme: item_rh,
                  bg: co,
                }

                l.push(n_item)
              })

              setLyrics(l)
              return
            }

            if (Object.prototype.hasOwnProperty.call(data, 'reason')) {
              setReasons(data.reason)
              return
            }

            if (Object.prototype.hasOwnProperty.call(data, 'pending')) {
              setPending(data.pending)
              return
            }
          } else {
            alert(res.data.msg)
          }
        }
      } catch (error) {
        alert(error)
      }
    }

    startFetching()

    return () => {
      ignore = true
    }
  }, [])

  const onNavList = () => {
    navigate(`/check/${token}?searchValue=${searchValue}`)
  }

  return (
    <div className="check_page">
      {songName !== '' ? <h2>歌名：{songName}</h2> : null}
      {songUrl !== '' ? (
        <div
          style={{ position: 'sticky', top: 0, backgroundColor: 'rgb(232, 237, 241)', zIndex: 1 }}
        >
          <AudioItem src={songUrl} mFirst={false} mToast={0} id={songName} />
        </div>
      ) : null}
      {lyrics.length > 0
        ? lyrics.map((item, index) => (
            <div
              key={index}
              style={{ display: 'flex', userSelect: 'none' }}
              className="check_show_lyrics_item"
            >
              <div style={{ fontSize: 'large' }} className="check_show_lyrics_item_type">
                <strong>{item.type}</strong>
              </div>
              <div style={{ fontSize: 'large' }}>{String(item.words).slice(0, -1)}</div>
              <div
                style={{
                  color: item.bg,
                  fontSize: 'large',
                  fontWeight: item.rhyme === undefined || item.rhyme === '' ? 'normal' : 'bold',
                }}
              >
                {String(item.words).slice(-1)}
              </div>
            </div>
          ))
        : null}
      {reasons.length > 0 ? (
        <div style={{ fontSize: 'large' }}>
          <strong>该歌曲被跳过</strong>
          <div>原因如下：</div>
          <ul>
            {reasons.map((item, index) => (
              <li style={{ margin: 0 }} key={index}>
                {item}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {pending !== '' ? (
        <div style={{ fontSize: 'large' }}>
          <strong>该歌曲被挂起</strong>
          <div>原因如下：</div>
          <div>{pending}</div>
        </div>
      ) : null}
      <Button type="primary" className="check_show_btn" onClick={onNavList}>
        回到当前列表
      </Button>
    </div>
  )
}

export default withRouter(CheckLyricShow)