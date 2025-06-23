import { Button, Input, Popconfirm, Popover, Select } from 'antd'
import Base64 from 'base-64'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { fetchLyricsAsync, pushLyricLansAsync, pushLyricsAsync } from '../../../api/lyrics'
import { getToken, getZjType, hasZjType, isQC, removeZjType, setAlert } from '../../../utils/auth'
import AudioItem from '../../Base/Audio'

const { TextArea } = Input
const Step2 = ({ id, type }) => {
  const [songName, setSongName] = useState('')
  const [songUrl, setSongUrl] = useState('')
  const [lyric, setLyric] = useState('')
  const [opts, setOpts] = useState([])
  const [selectOpts, setSelectOpts] = useState([])
  const [cantSub, setCantSub] = useState(true)

  const [confirmOpen, setConfirmOpen] = useState(false)

  const [loading, setLoading] = useState(false)

  const [tgLoading, setTgLoading] = useState(false)

  const params = {
    song_id: id,
    token: getToken(),
  }

  const navigate = useNavigate()

  useEffect(() => {
    window.addEventListener('beforeunload', beforeunload)
  }, [])

  const beforeunload = (ev) => {
    if (ev) {
      setAlert(false)
    }
  }

  useEffect(() => {
    let ignore = false

    const params = {
      song_id: id,
      token: getToken(),
    }

    async function startFetching() {
      try {
        const json = await fetchLyricsAsync(params, type)
        if (!ignore) {
          if (Number(json.data.code) === 200) {
            setSongName(json.data.data.name)
            setOpts(json.data.data.lans)
            setSongUrl(json.data.data.music)
            let lrc = ''
            if (Object.prototype.hasOwnProperty.call(json.data.data, 'lrc_list')) {
              json.data.data.lrc_list.forEach((item, index) => {
                lrc = index === 0 ? item : lrc + '\n' + item
              })
            } else {
              json.data.data.section_list.forEach((item, index) => {
                lrc = index === 0 ? item.words : lrc + '\n' + item.words
              })
            }
            setLyric(lrc)
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
  }, [id, type])

  const changeInputValue = (e) => {
    setLyric(e.target.value)
  }

  const getList = async () => {
    let str_sub = ''

    const item_list = lyric.split('\n')

    item_list.forEach((i) => {
      if (String(i) !== '' && String(i) !== '\n') {
        str_sub += `${i}\n`
      }
    })

    if (str_sub.substring(0, str_sub.length - 1) !== '') {
      const dict_sub = { song: str_sub.substring(0, str_sub.length - 1) }
      try {
        setLoading(true)
        const res = await pushLyricsAsync(params, dict_sub)
        if (res.data.code === 200) {
          if (isQC() === true) {
            const next_step = Base64.encode(`{"id": "${id}", "step": 4}`) 
            navigate(`/quality/${next_step}`)
          } else {
            const next_step = Base64.encode(`{"id": "${id}", "step": 3}`) 
            navigate(`/annotation/${next_step}`)
          }
        } else {
          alert(res.data.msg)
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
    } else {
      alert('请输入歌词！')
    }
  }

  const onClickPass = async () => {
    try {
      setTgLoading(true)
      const res = await pushLyricLansAsync(
        params,
        selectOpts,
        hasZjType() !== true ? '-1' : getZjType()
      )
      if (res.data.code === 200) {
        if (hasZjType() === true) {
          removeZjType()
        }
        setConfirmOpen(false)
        navigate('/home')
      } else {
        alert(res.data.msg)
        setTgLoading(false)
        return
      }
    } catch (error) {
      if (String(error).includes('timeout')) {
        alert('请求超时！')
      } else if (String(error).includes('500')) {
        alert('请求失败！')
      } else {
        alert(error)
      }
      setTgLoading(false)
      return
    }
  }

  const onClickCancel = () => {
    setConfirmOpen(false)
  }

  const changePass = (e) => {
    if (e.length > 0) {
      setCantSub(false)
    } else {
      setCantSub(true)
    }
    setSelectOpts(e)
  }

  const onPaste = (e) => {
    e.preventDefault()
    return false
  }

  const pop_content = (
    <div className="pop">
      <div>
        对歌词进行校验，下方的音频播放器可暂停/播放/拖拽进行操作，音频播放器下方是该首歌的歌词，需要听完整首歌曲来校对歌词文本内容，有的文本框内的第一句话是歌名/歌手名，不属于歌词内容，是需要删除的。
      </div>
      <div style={{ color: 'red' }}>注意：在这一步需要确保歌词的正确性及完整性！</div>
    </div>
  )

  return (
    <div style={{ width: '100%' }}>
      <div style={{ display: 'flex' }}>
        <h1>1/3 歌词校验</h1>
        <div>&nbsp;&nbsp;&nbsp;</div>
        <Popover content={pop_content}>
          <h1>ℹ️</h1>
        </Popover>
      </div>
      {songName !== '' ? <h2>歌名：{songName}</h2> : null}
      {songUrl !== '' ? (
        <div
          style={{ position: 'sticky', top: 0, backgroundColor: 'rgb(232, 237, 241)', zIndex: 1 }}
        >
          <div style={{ display: 'flex', alignItems: 'center' }} className="pass_div">
            <h3 style={{ margin: 0 }}>该歌曲需跳过，原因是：</h3>
            <Select
              mode="multiple"
              className="pass_sec"
              value={selectOpts}
              options={opts}
              onChange={(e) => changePass(e)}
            />
            <Popconfirm
              title="你确定这首歌需要跳过吗？"
              description="此过程不可逆！"
              open={confirmOpen}
              onConfirm={onClickPass}
              onCancel={onClickCancel}
            >
              <Button
                onClick={() => {
                  setConfirmOpen(true)
                }}
                type="primary"
                loading={tgLoading}
                className="pass_btn"
                disabled={cantSub}
              >
                确认并跳过
              </Button>
            </Popconfirm>
          </div>
          <AudioItem src={songUrl} mFirst={false} mToast={0} id={songName} />
        </div>
      ) : null}

      <TextArea
        type="text"
        className="lyrics_area"
        autoSize={true}
        value={lyric}
        onChange={(e) => changeInputValue(e)}
        onPaste={onPaste}
      />

      <Button type="primary" className="button" loading={loading} onClick={() => getList()}>
        提交
      </Button>
    </div>
  )
}

export default Step2