import { Button, Input } from 'antd'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { fetchMusicQusAsync, fetchMusicTagsAsync, pushMusicAnsAsync } from '../../../api/lyrics'
import { getFirst, getToken, removeFirst, setAlert } from '../../../utils/auth'
import AudioItem from '../../Base/Audio'
import EditC from './EditC'
import RadioC from './RadioC'
import RadioEditC from './RadioEditC'
import TagC from './TagC'
import TagEditC from './TagEditC'

const { TextArea } = Input

const Step1 = ({ id }) => {
  const initCanWrite = !(getFirst() === 'true')
  const [canWrite, setCanWrite] = useState(initCanWrite)

  const navigate = useNavigate()

  const [songName, setSongName] = useState('')
  const [songUrl, setSongUrl] = useState('')

  const [qusList, setQusList] = useState([])

  const [lrc, setLrc] = useState('')

  const [loading, setLoading] = useState(false)

  useEffect(() => {
    window.addEventListener('beforeunload', beforeunload)
  }, [])

  const beforeunload = (ev) => {
    const l_first = getFirst()
    if (ev) {
      if (l_first === 'true') {
        setAlert(true)
      }
      if (l_first === 'false') {
        setAlert(false)
      }
    }
  }

  useEffect(() => {
    if (!canWrite) {
      let time = 0
      const timer = setInterval(() => {
        if (time >= 30) {
          setCanWrite(true)
          clearInterval(timer)
        } else {
          time++
        }
      }, 1000)
    }
  }, [])

  useEffect(() => {
    let ignore = false

    const params = {
      song_id: id,
      token: getToken(),
    }

    async function startFetching() {
      try {
        const json = await fetchMusicQusAsync(params)
        if (!ignore) {
          if (Number(json.data.code) === 200) {
            setSongUrl(json.data.data.music)
            setSongName(json.data.data.name)
            setLrc(json.data.data.lrc)
            setQusList(json.data.data.qa)
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
    const l_first = getFirst()
    if (l_first === 'false') {
      const ans = []
      let canSub = true
      wai: for (let i = 0; i < qusList.length; i++) {
        const ans_item = { id: qusList[i].id }
        const tag_list = []
        const option_list = []
        for (let j = 0; j < qusList[i].ui.length; j++) {
          if (qusList[i].ui[j].type === '0') {
            if (qusList[i].ui[j].must === '0') {
              if (
                qusList[i].ui[j].desc !== null &&
                qusList[i].ui[j].desc !== undefined &&
                qusList[i].ui[j].desc !== ''
              ) {
                ans_item.desc = qusList[i].ui[j].desc
              } else {
                ans_item.desc = ''
              }
            } else {
              if (
                qusList[i].ui[j].desc === '' ||
                qusList[i].ui[j].desc === null ||
                qusList[i].ui[j].desc === undefined
              ) {
                canSub = false
                alert(`表单有非选填未填项（第 ${i + 1} 题），请填写完整`)
                break wai
              }
              if (qusList[i].ui[j].min_len !== -1) {
                if (String(qusList[i].ui[j].desc).length < qusList[i].ui[j].min_len) {
                  canSub = false
                  alert(`第 ${i + 1} 题中的非选填的描述项请至少输入 ${qusList[i].ui[j].min_len} 字`)
                  break wai
                }
              }
              ans_item.desc = qusList[i].ui[j].desc
            }
          }
          if (qusList[i].ui[j].type === '10') {
            if (
              qusList[i].ui[j].tValue === null ||
              qusList[i].ui[j].tValue === undefined ||
              qusList[i].ui[j].tValue.length === 0
            ) {
              canSub = false
              alert(`表单有非选填未填项（第 ${i + 1} 题），请填写完整`)
              break wai
            }
            if (qusList[i].ui[j].tValue.length < qusList[i].ui[j].min_tag) {
              canSub = false
              alert(`第 ${i + 1} 题中的Tag项请至少选择 ${qusList[i].ui[j].min_tag} 个`)
              break wai
            }
            const tag_item = {
              tid: qusList[i].ui[j].sub_id,
              value: qusList[i].ui[j].tValue,
            }
            tag_list.push(tag_item)
          }
          if (qusList[i].ui[j].type === '11') {
            if (
              qusList[i].ui[j].tValue === null ||
              qusList[i].ui[j].tValue === undefined ||
              qusList[i].ui[j].tValue.length === 0
            ) {
              canSub = false
              alert(`表单有非选填未填项（第 ${i + 1} 题），请填写完整`)
              break wai
            }
            if (qusList[i].ui[j].tValue.length < qusList[i].ui[j].min_tag) {
              canSub = false
              alert(`第 ${i + 1} 题中的Tag项请至少选择 ${qusList[i].ui[j].min_tag} 个`)
              break wai
            }
            if (
              qusList[i].ui[j].desc === '' ||
              qusList[i].ui[j].desc === null ||
              qusList[i].ui[j].desc === undefined
            ) {
              canSub = false
              alert(`表单有非选填未填项（第 ${i + 1} 题），请填写完整`)
              break wai
            }
            if (qusList[i].ui[j].min_len !== -1) {
              if (String(qusList[i].ui[j].desc).length < qusList[i].ui[j].min_len) {
                canSub = false
                alert(`第 ${i + 1} 题中的非选填的描述项请至少输入 ${qusList[i].ui[j].min_len} 字`)
                break wai
              }
            }

            const tag_item = {
              tid: qusList[i].ui[j].sub_id,
              value: qusList[i].ui[j].tValue,
              desc: qusList[i].ui[j].desc,
            }
            tag_list.push(tag_item)
          }
          if (qusList[i].ui[j].type === '20') {
            if (qusList[i].ui[j].rValue === null || qusList[i].ui[j].rValue === undefined) {
              canSub = false
              alert(`表单有非选填未填项（第 ${i + 1} 题），请填写完整`)
              break wai
            }
            const option_item = {}
            option_item.oid = qusList[i].ui[j].sub_id
            option_item.opid = qusList[i].ui[j].rValue
            option_list.push(option_item)
          }
          if (qusList[i].ui[j].type === '21') {
            if (qusList[i].ui[j].rValue === null || qusList[i].ui[j].rValue === undefined) {
              canSub = false
              alert(`表单有非选填未填项（第 ${i + 1} 题），请填写完整`)
              break wai
            }
            const option_item = {}
            option_item.oid = qusList[i].ui[j].sub_id
            option_item.opid = qusList[i].ui[j].rValue
            for (let k = 0; k < qusList[i].ui[j].options.length; k++) {
              if (qusList[i].ui[j].rValue === qusList[i].ui[j].options[k].id) {
                if (qusList[i].ui[j].options[k].must === '1') {
                  if (
                    qusList[i].ui[j].desc === null ||
                    qusList[i].ui[j].desc === undefined ||
                    qusList[i].ui[j].desc === ''
                  ) {
                    canSub = false
                    alert(`表单有非选填未填项（第 ${i + 1} 题），请填写完整`)
                    break wai
                  }
                  if (String(qusList[i].ui[j].desc).length < qusList[i].ui[j].options[k].min_len) {
                    canSub = false
                    alert(
                      `第 ${i + 1} 题中的非选填的描述项请至少输入 ${qusList[i].ui[j].options[k].min_len} 字`
                    )
                    break wai
                  }
                  option_item.desc = qusList[i].ui[j].desc
                } else {
                  if (
                    qusList[i].ui[j].desc !== null &&
                    qusList[i].ui[j].desc !== undefined &&
                    qusList[i].ui[j].desc !== ''
                  ) {
                    option_item.desc = qusList[i].ui[j].desc
                  }
                }
              }
            }
            option_list.push(option_item)
          }
        }
        if (tag_list.length !== 0) {
          ans_item.tags = tag_list
        }
        if (option_list.length !== 0) {
          ans_item.options = option_list
        }

        ans.push(ans_item)
      }

      if (canSub === true) {
        try {
          setLoading(true)

          const res = await pushMusicAnsAsync(id, getToken(), ans)
          if (res.data.code === 200) {
            removeFirst()
            navigate('/home')
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
      }
    } else {
      alert('请至少听完一遍')
    }
  }

  const onChangeSelector = (qus_id, ui_id, e) => {
    let ne = []
    if (e.length <= 5) {
      ne = e
    } else {
      ne = e.slice(0, 5)
    }
    qusList[qus_id].ui[ui_id].tValue = ne
    setQusList([...qusList])
  }

  const onFocus = async (ui_index, list_index) => {
    if (
      qusList[list_index].ui[ui_index].options === null ||
      qusList[list_index].ui[ui_index].options === undefined
    ) {
      try {
        const res = await fetchMusicTagsAsync(
          qusList[list_index].ui[ui_index].category_id,
          getToken()
        )
        if (res.data.code === 200) {
          const list = Array.from(new Set(res.data.data))

          qusList[list_index].ui[ui_index].options = list.map((item) => {
            return { value: item, label: item }
          })
          setQusList([...qusList])
        } else {
          alert(res.data.msg)
        }
      } catch (err) {
        alert(err)
      }
    }
  }

  const banPaste = (e) => {
    e.preventDefault()
    return false
  }

  const onChangeTextArea = (qus_id, ui_id, e) => {
    qusList[qus_id].ui[ui_id].desc = e.target.value
    setQusList([...qusList])
  }

  const onChangeRadio = (qus_id, ui_id, e) => {
    qusList[qus_id].ui[ui_id].rValue = e.target.value
    setQusList([...qusList])
  }

  const switchComp = (ui_item, ui_index, list_index) => {
    switch (qusList[list_index].ui[ui_index].type) {
      case '0':
        return (
          <EditC
            eValue={ui_item.desc}
            must={ui_item.must}
            minLen={ui_item.min_len}
            onPaste={(e) => banPaste(e)}
            onChange={(e) => onChangeTextArea(list_index, ui_index, e)}
          />
        )
      case '10':
        return (
          <TagC
            tValue={ui_item.tValue}
            options={ui_item.options}
            minTags={ui_item.min_tag}
            onFocus={() => onFocus(ui_index, list_index)}
            onChange={(e) => onChangeSelector(list_index, ui_index, e)}
          />
        )
      case '11':
        return (
          <TagEditC
            tValue={ui_item.tValue}
            eValue={ui_item.desc}
            minLen={ui_item.min_len}
            minTags={ui_item.min_tag}
            options={ui_item.options}
            onFocus={() => onFocus(ui_index, list_index)}
            onChangeE={(e) => onChangeTextArea(list_index, ui_index, e)}
            onChangeS={(e) => onChangeSelector(list_index, ui_index, e)}
            onPaste={(e) => banPaste(e)}
          />
        )
      case '20':
        return (
          <RadioC
            rValue={ui_item.rValue}
            options={ui_item.options}
            onChange={(e) => onChangeRadio(list_index, ui_index, e)}
          />
        )
      case '21':
        return (
          <RadioEditC
            rValue={ui_item.rValue}
            eValue={ui_item.desc}
            options={ui_item.options}
            onChangeR={(e) => onChangeRadio(list_index, ui_index, e)}
            onChangeE={(e) => onChangeTextArea(list_index, ui_index, e)}
            onPaste={(e) => banPaste(e)}
          />
        )
      default:
        return null
    }
  }

  return (
    <div>
      {songName !== '' ? <h2>歌名：{songName}</h2> : null}
      {songUrl !== '' ? (
        <div
          style={{ position: 'sticky', top: 0, backgroundColor: 'rgb(232, 237, 241)', zIndex: 1 }}
        >
          <AudioItem src={songUrl} mFirst={true} mToast={0} mPrompt={true} id={songName} />
        </div>
      ) : null}
      {songUrl !== '' ? (
        <div>
          <div className="qu_tip">1、进入页面30s后才能输入</div>
          <div className="qu_tip">
            2、整首歌完整播放完一遍后才能操作播放器（播放、暂停、进度条拖拽）
          </div>
          <div className="qu_tip">
            3、各个标签栏加入了联想功能，只需要打1个字/2个字就可以推荐热词库中的词，您来选择合适的标签，如果未推荐出您想要的标签，可以打字完成后按enter键创建新标签并输入
          </div>
        </div>
      ) : null}

      {songUrl !== '' ? (
        <div
          className="music_form"
          id="music_form"
          style={{ pointerEvents: `${canWrite ? 'auto' : 'none'}` }}
        >
          {qusList.map((item, index) => (
            <div key={item.id} className="qus">
              <h3>
                {index + 1}、{item.question}
              </h3>
              {item.ui.map((ui_item, ui_index) => {
                return (
                  <div key={`${item.id}_${ui_index}`}>
                    <h3>{ui_item.title}：</h3>
                    {ui_item.tips !== '' && ui_item.tips !== null && ui_item.tips !== undefined ? (
                      <div className="tips">{ui_item.tips}</div>
                    ) : null}
                    {switchComp(ui_item, ui_index, index)}
                  </div>
                )
              })}
            </div>
          ))}
          <Button type="primary" className="button" loading={loading} onClick={() => onSubmit()}>
            提交
          </Button>
          <TextArea
            type="text"
            style={{ textAlign: 'center' }}
            autoSize={{ maxRows: 10 }}
            value={lrc}
            overflow="true"
            disabled={false}
          />
        </div>
      ) : null}
    </div>
  )
}
export default Step1