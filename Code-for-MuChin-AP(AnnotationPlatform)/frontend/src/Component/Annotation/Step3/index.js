import { Button, Cascader, Popconfirm, Popover, Select } from 'antd'
import Base64 from 'base-64'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import {
  fetchSectionLyricsAsync,
  pushLyricLansAsync,
  pushSectionLyricsAsync,
} from '../../../api/lyrics'
import { getToken, getZjType, hasZjType, isQC, removeZjType } from '../../../utils/auth'
import AudioItem from '../../Base/Audio'

const Step3 = ({ id }) => {
  const [music, setMusic] = useState([])
  const [songName, setSongName] = useState('')
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
      try {
        const json = await fetchSectionLyricsAsync(params)
        if (!ignore) {
          if (Number(json.data.code) === 200) {
            setSongName(json.data.data.name)
            setSongUrl(json.data.data.music)
            setOpts(json.data.data.lans)
            if (
              isQC() === false &&
              !Object.prototype.hasOwnProperty.call(json.data.data.section_list[0], 'type')
            ) {
              const items = []
              json.data.data.section_list.forEach((item, index) => {
                const new_item = {
                  words: item.words,
                  type: undefined,
                  start: index,
                  end: index,
                  sp: false,
                }
                items.push(new_item)
                setMusic(items)
              })
            } else {
              const items = []
              let currentType = json.data.data.section_list[0].type
              let next_type_start = 0
              json.data.data.section_list.forEach((item, index) => {
                if (item.type === undefined) {
                  if (currentType !== undefined) {
                    items[index - 1].start = next_type_start
                    items[index - 1].sp = true
                    items[index - 1].type = [currentType.slice(0, -1), currentType.slice(-1)]
                    next_type_start = index
                    currentType = undefined
                  }

                  const new_item = {
                    words: item.words,
                    type: undefined,
                    start: index,
                    end: index,
                    sp: false,
                  }
                  items.push(new_item)
                } else {
                  if (currentType !== item.type) {
                    items[index - 1].start = next_type_start
                    items[index - 1].sp = true
                    items[index - 1].type = [currentType.slice(0, -1), currentType.slice(-1)]
                    next_type_start = index
                    currentType = item.type
                    if (index === json.data.data.section_list.length - 1) {
                      const new_item = {
                        words: item.words,
                        type: [item.type.slice(0, -1), item.type.slice(-1)],
                        start: index,
                        end: index,
                        sp: true,
                      }
                      items.push(new_item)
                    } else {
                      const new_item = {
                        words: item.words,
                        type: undefined,
                        start: index,
                        end: index,
                        sp: false,
                      }
                      items.push(new_item)
                    }
                  } else {
                    if (index === json.data.data.section_list.length - 1) {
                      const new_item = {
                        words: item.words,
                        type: [currentType.slice(0, -1), currentType.slice(-1)],
                        start: next_type_start,
                        end: index,
                        sp: true,
                      }
                      items.push(new_item)
                    } else {
                      const new_item = {
                        words: item.words,
                        type: undefined,
                        start: index,
                        end: index,
                        sp: false,
                      }
                      items.push(new_item)
                    }
                  }
                }
              })

              setMusic(items)
            }
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

  const selectChange = (value, index) => {
    music[index].type = value

    setMusic([...music])
  }

  const gotoPre = async () => {
    if (isQC === true) {
      const pre_step = Base64.encode(`{"id": "${id}", "step": 3, "type": 1}`)
      navigate(`/quality/${pre_step}`)
    } else {
      const pre_step = Base64.encode(`{"id": "${id}", "step": 2, "type": 1}`)
      navigate(`/annotation/${pre_step}`)
    }
  }

  const getList = async () => {
    const section = []
    let sub = false

    wai: for (let i = 0; i < music.length; i++) {
      const item = music[i]
      if (item.sp === true) {
        sub = true
        for (let j = 0; j <= item.end - item.start; j++) {
          if (item.type === undefined || item.type === '') {
            sub = false
            break wai
          }
          section.push(`${item.type[0]}${item.type[1]}`)
        }
      }
    }

    if (sub === true && music.slice(-1)[0].sp !== false) {
      try {
        setLoading(true)
        const res = await pushSectionLyricsAsync(params, section)
        if (res.data.code === 200) {
          if (isQC() === true) {
            const next_step = Base64.encode(`{"id": "${id}", "step": 5}`)
            navigate(`/quality/${next_step}`)
          } else {
            const next_step = Base64.encode(`{"id": "${id}", "step": 4}`)
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
      alert('请给歌词标段落！')
    }
  }

  const onClickSplit = (id, sp) => {
    music[id].type = undefined
    music[id].sp = sp !== true ? true : false

    if (music.length !== 1) {
      if (sp !== true) {
        if (id === 0) {
          for (let i = id + 1; i < music.length; i++) {
            if (music[i].sp === true) {
              music[i].start = id + 1
              break
            }
          }
          music[id].start = id
          music[id].end = id
        } else if (id === music.length - 1) {
          let index = 0

          for (let i = id - 1; i >= 0; i--) {
            if (music[i].sp === true) {
              index = music[i].end + 1
              break
            }
          }

          music[id].start = index
          music[id].end = id
        } else {
          for (let i = id + 1; i < music.length; i++) {
            if (music[i].sp === true) {
              music[i].start = id + 1
              break
            }
          }

          let index = 0
          let hasSp = false
          for (let i = id - 1; i >= 0; i--) {
            if (music[i].sp === true) {
              index = music[i].end + 1
              hasSp = true
              break
            }
          }

          music[id].start = hasSp === false ? 0 : index
          music[id].end = id
        }
      } else {
        if (id === 0) {
          for (let i = id + 1; i < music.length; i++) {
            if (music[i].sp === true) {
              music[i].start = id
              break
            }
          }
          music[id].start = id
          music[id].end = id
        } else if (id === music.length - 1) {
          music[id].start = id
          music[id].end = id
        } else {
          for (let i = id + 1; i < music.length; i++) {
            if (music[i].sp === true) {
              music[i].start = music[id].start
              break
            }
          }
          music[id].start = id
          music[id].end = id
        }
      }
    }
    setMusic([...music])
  }

  const ItemUi = (item, index) => {
    return (
      <div key={index}>
        <div className="sec_item_ui" style={{ display: 'flex', alignItems: 'center' }}>
          <div className="section_item" style={{ flex: 1 }} id={index}>
            {item.words}
          </div>

          <div
            onClick={() => onClickSplit(index, item.sp)}
            style={{ cursor: 'pointer', fontSize: 'large', userSelect: 'none' }}
          >
            {item.sp !== true ? '➕' : '➖'}
          </div>
        </div>
        {item.sp === true ? (
          <div style={{ display: 'flex', alignItems: 'center' }} className="sec_item_ui">
            <hr style={{ flex: 1 }} />
            <div style={{ fontSize: 'large', fontWeight: 'bold' }} className="sec_item_tip">
              以上为：
            </div>
            <Cascader
              className="sec_item_cas"
              value={item.type}
              onChange={(value) => selectChange(value, index)}
              options={tagOptions}
              displayRender={(label) => label}
            />
          </div>
        ) : null}
      </div>
    )
  }
  const tagOptions = [
    {
      value: '主歌',
      label: '主歌',
      children: [
        { value: '1', label: '1' },
        { value: '2', label: '2' },
        { value: '3', label: '3' },
        { value: '4', label: '4' },
        { value: '5', label: '5' },
        { value: '6', label: '6' },
        { value: '7', label: '7' },
        { value: '8', label: '8' },
        { value: '9', label: '9' },
      ],
    },
    {
      value: '副歌',
      label: '副歌',
      children: [
        { value: '1', label: '1' },
        { value: '2', label: '2' },
        { value: '3', label: '3' },
        { value: '4', label: '4' },
        { value: '5', label: '5' },
        { value: '6', label: '6' },
        { value: '7', label: '7' },
        { value: '8', label: '8' },
        { value: '9', label: '9' },
      ],
    },
    {
      value: '预副',
      label: '预副',
      children: [
        { value: '1', label: '1' },
        { value: '2', label: '2' },
        { value: '3', label: '3' },
        { value: '4', label: '4' },
        { value: '5', label: '5' },
        { value: '6', label: '6' },
        { value: '7', label: '7' },
        { value: '8', label: '8' },
        { value: '9', label: '9' },
      ],
    },
    {
      value: '桥段',
      label: '桥段',
      children: [
        { value: '1', label: '1' },
        { value: '2', label: '2' },
        { value: '3', label: '3' },
        { value: '4', label: '4' },
        { value: '5', label: '5' },
        { value: '6', label: '6' },
        { value: '7', label: '7' },
        { value: '8', label: '8' },
        { value: '9', label: '9' },
      ],
    },
    {
      value: '前奏',
      label: '前奏',
      children: [
        { value: '1', label: '1' },
        { value: '2', label: '2' },
        { value: '3', label: '3' },
        { value: '4', label: '4' },
        { value: '5', label: '5' },
        { value: '6', label: '6' },
        { value: '7', label: '7' },
        { value: '8', label: '8' },
        { value: '9', label: '9' },
      ],
    },
    {
      value: '间奏',
      label: '间奏',
      children: [
        { value: '1', label: '1' },
        { value: '2', label: '2' },
        { value: '3', label: '3' },
        { value: '4', label: '4' },
        { value: '5', label: '5' },
        { value: '6', label: '6' },
        { value: '7', label: '7' },
        { value: '8', label: '8' },
        { value: '9', label: '9' },
      ],
    },
    {
      value: '尾奏',
      label: '尾奏',
      children: [
        { value: '1', label: '1' },
        { value: '2', label: '2' },
        { value: '3', label: '3' },
        { value: '4', label: '4' },
        { value: '5', label: '5' },
        { value: '6', label: '6' },
        { value: '7', label: '7' },
        { value: '8', label: '8' },
        { value: '9', label: '9' },
      ],
    },
  ]

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

  const pop_content = (
    <div className="pop">
      <div>对每一句歌词进行段落类型标注，段落标注的结果将会影响下一步押韵信息的校验</div>
      <div style={{ color: 'red' }}>需要保证段落标注的准确性！</div>
      <ul color="red">
        <li>主歌（verse）</li>
        <li>预副（也叫导歌，pre-chorus）</li>
        <li>副歌（也叫复歌，chorus）</li>
        <li>桥段（bridge）</li>
        <li>前奏（也叫引子，prelude，intro，一般无歌词）</li>
        <li>间奏（interlude，一般无歌词）</li>
        <li>尾奏（也叫尾声，ending，outro，一般无歌词）</li>
      </ul>
    </div>
  )

  return (
    <div>
      <div style={{ display: 'flex' }}>
        <h1>2/3 段落标注</h1>
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
      {music.map((item, index) => ItemUi(item, index))}
      {songUrl !== '' ? (
        <Button type="primary" className="button" disabled={loading} onClick={() => gotoPre()}>
          上一步
        </Button>
      ) : null}
      {songUrl !== '' ? (
        <Button type="primary" className="button" loading={loading} onClick={() => getList()}>
          提交
        </Button>
      ) : null}
    </div>
  )
}
export default Step3