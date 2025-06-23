import { Button, Input, Popconfirm, Popover, Select } from 'antd'
import Base64 from 'base-64'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { pushCheckSubmitAsync } from '../../../api/check'
import {
  fetchRhymeLyricsAsync,
  pushLyricLansAsync,
  pushRhymeEnterAsync,
  pushRhymeLyricsAsync,
} from '../../../api/lyrics'
import { getToken, getZjType, hasZjType, isQC, removeZjType } from '../../../utils/auth'
import initColors from '../../../utils/colors'
import AudioItem from '../../Base/Audio'

const Step4 = ({ id }) => {
  const [songName, setSongName] = useState('')
  const [list, setList] = useState([])
  const [uc, setUc] = useState(initColors)
  const [songUrl, setSongUrl] = useState('')
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
    let ignore = false

    const params = {
      song_id: id,
      token: getToken(),
    }

    async function startFetching() {
      const lyrics = []
      try {
        const json = await fetchRhymeLyricsAsync(params)
        if (!ignore) {
          if (Number(json.data.code) === 200) {
            let temp_uc = uc

            let index = 0
            let ttype = ''
            const map = new Map()
            json.data.data.rhyme_list.forEach((item) => {
              const il = item.line
              const t = item.type

              if (il === 0) {
                ttype = t
              }
              if (t !== ttype) {
                ttype = t
                map.clear()
              }
              let co = ''
              const rh = item.rhyme
              if (Object.prototype.hasOwnProperty.call(item, 'rhyme')) {
                if (!map.has(rh)) {
                  const ic = uc[index]
                  map.set(rh, ic)

                  temp_uc = temp_uc.filter((temp_uc_item) => temp_uc_item !== ic)

                  index++
                }
                co = map.get(rh)
              } else {
                co = ''
              }
              const n_item = {
                line: il,
                type: ttype,
                words: item.words,
                rhyme: rh,
                pinyin: item.pinyin,
                bg: co,
              }
              lyrics.push(n_item)
            })
            setList(lyrics)
            setSongName(json.data.data.name)
            setSongUrl(json.data.data.music)
            setUc(temp_uc)
            setOpts(json.data.data.lans)
          } else {
            alert(json.data.msg)
          }
        }
      } catch (err) {
        alert(err)
      }
    }
    startFetching()
    return () => {
      ignore = true
    }
  }, [id])

  const onSubmit = async () => {
    if (list.length > 0) {
      const l = []
      list.forEach((item) => {
        l.push(item.rhyme === undefined ? '' : item.rhyme)
      })

      if (l.length > 0) {
        try {
          setLoading(true)
          let res
          if (isQC() === true) {
            res = await pushCheckSubmitAsync(params, l, hasZjType() !== true ? '-1' : getZjType())
          } else {
            res = await pushRhymeLyricsAsync(params, l)
          }
          if (res.data.code === 200) {
            if (hasZjType() === true) {
              removeZjType()
            }
            navigate('/home')
          } else {
            alert(res.data.msg)
            setLoading(false)
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
          setLoading(false)
          return
        }
      } else {
        alert('数据异常')
      }
    } else {
      alert('数据异常')
    }
  }

  const onChange = (id, e) => {
    const v = e.target.value
    const reg = /^[a-z]+$/
    if (reg.exec(String(v).trim()) || String(v).trim() === '') {
      list[id].pinyin = v
      setList([...list])
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

  const onPressEnter = async (id) => {
    const l = []
    let start = -1
    let end = -1
    const tp = list[id].type
    const nc = []
    const tttnc = uc

    for (let i = id; i >= 0; i--) {
      if (i === 0) {
        if (list[i].type === tp) {
          start = 0
        }
      }
      if (list[i].type !== tp) {
        start = i + 1
        break
      }
      l.unshift({
        words: list[i].words,
        pinyin: list[i].pinyin === undefined ? '' : list[i].pinyin,
      })
      if (nc.indexOf(list[i].bg) === -1 && list[i].bg !== '') {
        nc.unshift(list[i].bg)
      }
    }
    if (id + 1 === list.length) {
      end = id
    } else {
      for (let i = id + 1; i < list.length; i++) {
        if (i === list.length - 1) {
          if (list[i].type === tp) {
            end = list.length - 1
          }
        }
        if (list[i].type !== tp) {
          end = i - 1
          break
        }
        l.push({
          words: list[i].words,
          pinyin: list[i].pinyin === undefined ? '' : list[i].pinyin,
        })
        if (nc.indexOf(list[i].bg) === -1 && list[i].bg !== '') {
          nc.push(list[i].bg)
        }
      }
    }

    nc.reverse().forEach((it) => {
      tttnc.unshift(it)
    })

    try {
      const res = await pushRhymeEnterAsync(params.token, l)
      if (res.data.code === 200) {
        const map = new Map()
        let j = 0
        let k = 0
        let temp_uc = tttnc
        for (let i = start; i <= end; i++) {
          list[i].pinyin = res.data.data[j].pinyin
          const rh = res.data.data[j].rhyme
          list[i].rhyme = rh
          if (rh === undefined) {
            list[i].bg = ''
          } else {
            if (!map.has(rh)) {
              const old = uc[k]
              map.set(rh, old)
              temp_uc = temp_uc.filter((temp_uc_item) => temp_uc_item !== old)
              k++
            }
            list[i].bg = map.get(rh)
          }
          j++
        }
        setList([...list])
        setUc(temp_uc)
      } else {
        alert(res.data.msg)
      }
    } catch (error) {
      alert(error)
    }
  }

  const gotoPre = async () => {
    if (isQC() === true) {
      const pre_step = Base64.encode(`{"id": "${id}", "step": 4}`)
      navigate(`/quality/${pre_step}`)
    } else {
      const pre_step = Base64.encode(`{"id": "${id}", "step": 3}`)
      navigate(`/annotation/${pre_step}`)
    }
  }

  const pop_content = (
    <div className="pop">
      <div>系统会自动以相同颜色高亮标出句尾相同押韵的字，基本只需检查多音字的发音。</div>
      <div style={{ color: 'red' }}>输入拼音后请按回车键，这会影响到押韵标注的准确性！</div>
    </div>
  )

  return (
    <div>
      <div style={{ display: 'flex' }}>
        <h1>3/3 押韵标注</h1>
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

      <ul style={{ padding: 0 }}>
        {list.map((item) => {
          return (
            <li className="rhyme_li" key={item.line}>
              <div>
                {item.type}：{String(item.words).slice(0, -1)}
              </div>
              <div
                className="rhyme_ly"
                style={{
                  color: item.bg,
                  fontWeight: item.rhyme === undefined || item.rhyme === '' ? 'normal' : 'bold',
                }}
              >
                {String(item.words).slice(-1)}
              </div>
              <Input
                className="rhyme_input"
                id={item.line}
                value={item.pinyin}
                size="small"
                onChange={(e) => onChange(item.line, e)}
                onPressEnter={() => onPressEnter(item.line)}
              />
            </li>
          )
        })}
      </ul>
      {songUrl !== '' ? (
        <Button type="primary" className="button" disabled={loading} onClick={() => gotoPre()}>
          上一步
        </Button>
      ) : null}
      {songUrl !== '' ? (
        <Button type="primary" className="button" loading={loading} onClick={() => onSubmit()}>
          提交
        </Button>
      ) : null}
    </div>
  )
}
export default Step4