import { Button, Popconfirm, Popover, Select } from 'antd'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { pushCheckSubmitAsync2 } from '../../../api/check'
import {
  fetchLyricsAsync2,
  pushLyricLansAsync,
  pushMusicInfoAsync,
  pushRhymeEnterAsync,
} from '../../../api/lyrics'
import { getToken, getZjType, hasZjType, removeZjType } from '../../../utils/auth'
import initColors from '../../../utils/colors'
import AudioItem from '../../Base/Audio'
import LyricsItemC from './LyricsItem'

const Lyrics = ({ id, type }) => {
  const [lyric, setLyric] = useState([])
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
  const charColors = initColors.map((color, index) => {
    return { char: String.fromCharCode(65 + index), color: color }
  })
  const navigate = useNavigate()
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

  const pop_content = (
    <div className="pop">
      <div>乐段标注，下方的音频播放器可暂停/播放/拖拽进行操作。</div>
    </div>
  )

  useEffect(() => {
    let ignore = false

    const params = {
      song_id: id,
      token: getToken(),
    }

    async function startFetching() {
      try {
        let get_type
        if (type === 0 || type === 1) {
          get_type = 0
        } else {
          get_type = 1
        }
        const json = await fetchLyricsAsync2(params, get_type)
        if (!ignore) {
          if (Number(json.data.code) === 200) {
            setSongName(json.data.data.name)
            setOpts(json.data.data.lans)
            setSongUrl(json.data.data.music)
            const list = setInfoData(json.data.data.info)
            setLyric(list)
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
  }, [id])

  const setInfoData = (info) => {
    const nestSpIndex = []
    let lastType = ''
    const list = info.map((item, index) => {
      const char = item.rhyme
      let bg = ''
      if (char === undefined || char === '') {
        bg = '#000000'
      } else {
        const is_char = (element) => element.char === char
        const char_index = charColors.findIndex(is_char)
        bg = charColors[char_index].color
      }
      if (lastType !== '' && lastType !== item.type) {
        nestSpIndex.push(index)
      }
      lastType = item.type
      return {
        ...item,
        pinyin_change: '',
        time_change: '',
        sp: false,
        bg: bg,
      }
    })
    nestSpIndex.push(list.length)
    nestSpIndex.map((index) => {
      list[index - 1].sp = true
    })
    return list
  }

  const getLyricsRhyme = async (index) => {
    let top_index = 0
    let bottom_index = lyric.length - 1

    const sp_list = lyric
      .map((item, index) => {
        return { words: item.words, pinyin_change: item.pinyin_change, sp: item.sp, index: index }
      })
      .filter((item) => {
        return item.sp === true
      })
    for (const index_sp in sp_list) {
      const index_sp_index = sp_list[index_sp].index
      if (index_sp_index >= index) {
        bottom_index = index_sp_index
        break
      } else {
        top_index = index_sp_index + 1
      }
    }

    var target_lyric = lyric.slice(top_index, bottom_index + 1)
    let has_null = false
    const rhyme_list = target_lyric.map((item) => {
      if (item.words === '') {
        has_null = true
      }
      return {
        words: item.words,
        pinyin: item.pinyin_change === undefined ? '' : item.pinyin_change,
      }
    })
    if (has_null) {
      alert('请将歌词填写完整')
      return
    }
    try {
      const res = await pushRhymeEnterAsync(params.token, rhyme_list)
      if (res.data.code === 200) {
        const change_lyric = res.data.data
        target_lyric.forEach((item, index) => {
          item.pinyin = change_lyric[index].pinyin
          item.rhyme = change_lyric[index].rhyme
          const char = change_lyric[index].rhyme
          if (char === undefined) {
            item.bg = '#000000'
          } else {
            const is_char = (element) => element.char === char
            const char_index = charColors.findIndex(is_char)
            item.bg = charColors[char_index].color
          }
        })
        lyric.splice(top_index, target_lyric.length, ...target_lyric)
        setLyric([...lyric])
      } else {
        alert(res.data.msg)
      }
    } catch (error) {
      alert(error)
    }
  }

  const onLyricsInputChange = (e, index) => {
    const origin_words = lyric[index].words
    const origin_words_last = origin_words.charAt(origin_words.length - 1)
    const change_words = e.target.value
    const change_words_last = change_words.charAt(change_words.length - 1)
    if (origin_words_last !== change_words_last) {
      lyric[index].pinyin_change = ''
    }
    lyric[index].words = e.target.value
    setLyric([...lyric])
  }

  const onLyricsPressEnter = (index) => {
    getLyricsRhyme(index)
  }

  const onPinyinInputChange = (e, index) => {
    lyric[index].pinyin = e.target.value
    lyric[index].pinyin_change = e.target.value
    setLyric([...lyric])
  }
  const onPinyinPressEnter = async (index) => {
    getLyricsRhyme(index)
  }
  const onTimeInputChange = (e, index) => {
    lyric[index].time_change = e.target.value
    setLyric([...lyric])
    console.log('onTimeInputChange' + index)
  }
  const onTimePressEnter = async (index) => {
    const text = lyric[index].time_change
    var patt = /^(\d{2}):([0-5][0-9]).(\d{2})$/
    if (!patt.test(text)) {
      lyric[index].time_change = ''
      setLyric([...lyric])
    } else {
      lyric[index].time = text
      lyric[index].time_change = ''
      setLyric([...lyric])
    }
    console.log('onTimePressEnter' + index)
  }

  const onAddLine = (index) => {
    const type = lyric[index].type
    const index_sp = lyric[index].sp
    let add_sp = false
    if (index_sp === true) {
      lyric[index].sp = false
      add_sp = true
    }

    const add_obj = {
      words: '',
      type: type,
      bg: '#000000',
      pinyin: '',
      time: '',
      time_change: '',
      sp: add_sp,
    }
    lyric.splice(index + 1, 0, add_obj)
    setLyric([...lyric])
  }

  const onRemoveLine = (index) => {
    if ((index > 0) & (lyric[index].sp === true)) {
      lyric[index - 1].sp = true
    }
    lyric.splice(index, 1)
    setLyric([...lyric])
  }

  const onClickSplit = (index) => {
    if (lyric[index].sp === true) {
      lyric[index].sp = false

      if (index === lyric.length - 1) {
        setLyric([...lyric])
        return
      }

      const target_type = lyric[index + 1].type

      var lyric_sp = lyric.slice(0, index).filter((item) => {
        return item.sp === true
      })
      var top_index = lyric.lastIndexOf(lyric_sp.pop(), index) + 1
      var change_lyric = lyric.slice(top_index, index + 1)
      change_lyric.forEach((item) => {
        item.type = target_type
      })
      lyric.splice(top_index, change_lyric.length, ...change_lyric)
      setLyric([...lyric])
      getLyricsRhyme(index)
    } else {
      lyric[index].sp = true
      setLyric([...lyric])
      getLyricsRhyme(index)
      getLyricsRhyme(index + 1)
    }
  }

  const selectChange = (index, value) => {
    var lyric_sp = lyric.slice(0, index).filter((item) => {
      return item.sp === true
    })
    var top_index = lyric.lastIndexOf(lyric_sp.pop(), index) + 1

    var change_lyric = lyric.slice(top_index, index + 1)
    change_lyric.forEach((item) => {
      item.type = value
    })
    lyric.splice(top_index, change_lyric.length, ...change_lyric)
    setLyric([...lyric])
  }

  const changePass = (e) => {
    if (e.length > 0) {
      setCantSub(false)
    } else {
      setCantSub(true)
    }
    setSelectOpts(e)
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

  const checkTimeAscending = (lastTime, currentTime) => {
    const time_last = lastTime.split(':')
    const time_current = currentTime.split(':')
    if (parseInt(time_current[0]) > parseInt(time_last[0])) {
      return true
    }
    if (parseFloat(time_current[1]) > parseFloat(time_last[1])) {
      return true
    }
    return false
  }

  const getList = async () => {
    setLoading(true)

    if (lyric.length === 0) {
      alert('无法提交空乐段，如果该首歌曲确实没有歌词，请跳过')
      setLoading(false)
      return
    }
    let words_null = false
    let type_null = false
    let time_null = false
    let time_last_record = ''
    let time_check = true
    let time_asc = true
    var patt = /^(\d{2}):([0-5][0-9]).(\d{2})$/

    const param = lyric.map((item) => {
      if (item.words === '') {
        words_null = true
      }
      if (item.type === '') {
        type_null = true
      }
      if (item.type === '') {
        type_null = true
      }
      if (item.time === '') {
        time_null = true
      }
      if (!patt.test(item.time)) {
        time_check = false
      }
      if (time_last_record.length > 0 && item.time.length > 0) {
        if (checkTimeAscending(time_last_record, item.time) === false) {
          time_asc = false
        }
      }
      time_last_record = item.time
      let type_string = ''
      if (typeof item.type === 'string') {
        type_string = item.type
      } else {
        type_string = item.type.join().replace(',', '')
      }
      return {
        words: item.words,
        type: type_string,
        rhyme: item.rhyme,
        pinyin: item.pinyin_change === undefined ? '' : item.pinyin_change,
        time: item.time,
      }
    })
    if (words_null) {
      alert('请将歌词填写完整')
      setLoading(false)
      return
    }
    if (type_null) {
      alert('歌词有未标段落，请给所有歌词标段落')
      setLoading(false)
      return
    }
    if (time_null) {
      alert('歌词有未标时间，请给所有歌词标时间')
      setLoading(false)
      return
    }
    if (!time_check) {
      alert('歌词时间戳格式应为mm:ss.xx，请检查修改后提交')
      setLoading(false)
      return
    }
    if (!time_asc) {
      alert('歌词时间戳需升序，请检查修改后提交')
      setLoading(false)
      return
    }
    const json_string = JSON.stringify(param)
    console.log(json_string)

    try {
      let res
      if (type === 0) {
        res = await pushMusicInfoAsync({ song_id: id, token: getToken() }, param)
      } else {
        res = await pushCheckSubmitAsync2(
          { song_id: id, token: getToken() },
          param,
          hasZjType() !== true ? '-1' : getZjType()
        )
      }
      if (res.data.code === 200) {
        setLoading(false)
        if (type !== 0) {
          if (hasZjType() === true) {
            removeZjType()
          }
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
  }

  return (
    <div>
      <div style={{ display: 'flex' }}>
        <h1>乐段标注</h1>
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
                className="pass_btn"
                loading={tgLoading}
                disabled={cantSub}
              >
                确认并跳过
              </Button>
            </Popconfirm>
          </div>
          <AudioItem src={songUrl} mFirst={false} mToast={0} id={songName} />
        </div>
      ) : null}

      {lyric.map((item, index) => {
        return (
          <LyricsItemC
            key={index}
            item={item}
            tagOptions={tagOptions}
            onLyricsInputChange={(e) => {
              onLyricsInputChange(e, index)
            }}
            onLyricsPressEnter={() => onLyricsPressEnter(index)}
            onPinyinInputChange={(e) => onPinyinInputChange(e, index)}
            onPinyinPressEnter={() => onPinyinPressEnter(index)}
            onTimeInputChange={(e) => onTimeInputChange(e, index)}
            onTimePressEnter={() => onTimePressEnter(index)}
            onAddLine={() => onAddLine(index)}
            onRemoveLine={() => onRemoveLine(index)}
            onClickSplit={() => onClickSplit(index)}
            selectChange={(value) => selectChange(index, value)}
          />
        )
      })}
      {songUrl !== '' ? (
        <Button type="primary" className="button" loading={loading} onClick={() => getList()}>
          提交
        </Button>
      ) : null}
    </div>
  )
}

export default Lyrics