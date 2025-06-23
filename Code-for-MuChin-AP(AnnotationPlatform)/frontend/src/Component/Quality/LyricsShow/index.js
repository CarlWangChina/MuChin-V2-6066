import '../index.scss'

import { Button } from 'antd'
import Base64 from 'base-64'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { fetchCheckLyricsAsync, pushCheckSelectedAsync } from '../../../api/check'
import { getToken, setZjType } from '../../../utils/auth'
import initColors from '../../../utils/colors'
import AudioItem from '../../Base/Audio'
import LrcShowC from './LrcShowC'

const LyricsShow = ({ id, work_id }) => {
  const [lyrics, setLyrics] = useState([])

  const [songName, setSongName] = useState('')

  const [songUrl, setSongUrl] = useState('')

  const [bzCount, setBzCount] = useState(0)
  const [tgCount, setTgCount] = useState(0)

  const [loading, setLoading] = useState(false)

  const navigate = useNavigate()

  useEffect(() => {
    const smallScreen = window.matchMedia('(max-width:768px)')
    const check_show = document.getElementById('check_show')
    if (smallScreen.matches) {
      check_show.style.flexDirection = 'column'
    } else {
      check_show.style.flexDirection = 'row'
    }
  })

  useEffect(() => {
    let ignore = false

    const params = {
      song_id: work_id,
      token: getToken(),
    }

    async function startFetching() {
      try {
        const json = await fetchCheckLyricsAsync(params)
        if (!ignore) {
          if (Number(json.data.code) === 200) {
            setSongUrl(json.data.data.music)
            setSongName(json.data.data.name)

            const list = []
            json.data.data.compare_info.forEach((compare_info) => {
              if (compare_info.info.length > 0) {
                setBzCount((bzCount) => (bzCount += 1))
                const new_info = []
                let currentType
                const map = new Map()
                let i = 0

                compare_info.info.forEach((item, index) => {
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
                  new_info.push(n_item)
                })

                const new_obj = { id: compare_info.id, user: compare_info.user, info: new_info }
                list.push(new_obj)
              } else {
                setTgCount((tgCount) => (tgCount += 1))
                const new_obj = {
                  id: compare_info.id,
                  user: compare_info.user,
                  info: [],
                  lans: compare_info.lans,
                }
                list.push(new_obj)
              }
            })
            setLyrics(list)
          } else {
            alert(json.data.msg)
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
  }, [work_id])

  const onSubmit = async (type, index) => {
    let tip = ''

    const error_list = []
    const right_list = []
    if (type === 0) {
      lyrics.forEach((item, i) => {
        if (i !== index) {
          error_list.push(item.id)
        }
      })
      error_list.unshift(lyrics[index].id)
      setZjType(type)
      tip = '你确定要选择对该结果进行修改嘛？\n该操作会在被选那份的基础上进行修正，两份都记为错。'
    }
    if (type === 1) {
      lyrics.forEach((item, i) => {
        if (i !== index) {
          error_list.push(item.id)
        }
      })
      right_list.unshift(lyrics[index].id)
      tip =
        '你确定要选择该结果为正确版本嘛？\n该操作会将被选的那份明确记为正确，另一份明确记为错误。'
    }
    if (type === 2) {
      lyrics.forEach((item) => {
        right_list.push(item.id)
      })
      setZjType(type)
      tip = '你确定要豁免二者并重新标嘛？\n该操作需要质检员开始像标注人员一样对这首歌从头开始标。'
    }

    const pass = window.confirm(tip)

    if (pass === true) {
      setLoading(true)
      const res_obj = { error: error_list, right: right_list }
      try {
        const json = await pushCheckSelectedAsync(id, getToken(), res_obj)

        if (Number(json.data.code) === 200) {
          const res = json.data.data.return
          if (res === 1) {
            navigate('/home')
          } else {
            if (type === 2 || json.data.data.lan === 1) {
              const next_step = Base64.encode(`{"id": "${id}", "step": 3, "type": 1}`)
              navigate(`/quality/${next_step}`)
            } else {
              const next_step = Base64.encode(`{"id": "${id}", "step": 3, "type": 2}`)
              navigate(`/quality/${next_step}`)
            }
          }
        } else {
          alert(json.data.msg)
          setLoading(false)
          return
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
  }

  return (
    <div>
      {songName !== '' ? <h2>歌名：{songName}</h2> : null}
      {songUrl !== '' ? (
        <div
          style={{ position: 'sticky', top: 0, backgroundColor: 'rgb(232, 237, 241)', zIndex: 1 }}
        >
          <AudioItem src={songUrl} mFirst={false} mToast={0} id={songName} />
        </div>
      ) : null}

      {bzCount > 0 && tgCount > 0 ? null : (
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <Button type="primary" className="long_btn" loading={loading} onClick={() => onSubmit(2)}>
            豁免二者并重新标
          </Button>
          <div className="check_btn_tip">质检员开始像标注人员一样对这首歌从头开始标</div>
        </div>
      )}

      <div style={{ display: 'flex' }} id="check_show">
        {lyrics.map((item, index) => (
          <LrcShowC
            key={item.id}
            data={item}
            loading={loading}
            onSubmitRight={() => onSubmit(1, index)}
            onSubmitError={() => onSubmit(0, index)}
          />
        ))}
      </div>
    </div>
  )
}

export default LyricsShow