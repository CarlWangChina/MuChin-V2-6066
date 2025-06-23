import { Button, Input, InputNumber } from 'antd'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { fetchCheckMusicAsync, pushCheckMusicScoreAsync } from '../../../api/check'
import { getFirst, getToken, setAlert } from '../../../utils/auth'
import EditC from '../../Annotation/Step1/EditC'
import RadioC from '../../Annotation/Step1/RadioC'
import RadioEditC from '../../Annotation/Step1/RadioEditC'
import TagC from '../../Annotation/Step1/TagC'
import TagEditC from '../../Annotation/Step1/TagEditC'
import AudioItem from '../../Base/Audio'

const { TextArea } = Input

const MusicShow = ({ work_id }) => {
  const initCanWrite = !(getFirst() === 'true')
  const [canWrite, setCanWrite] = useState(initCanWrite)

  const navigate = useNavigate()

  const [songName, setSongName] = useState('')
  const [songUrl, setSongUrl] = useState('')

  const [qusList, setQusList] = useState([])

  const [lrc, setLrc] = useState('')

  const [loading, setLoading] = useState(false)

  const [allScore, setAllScore] = useState(0)

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
      song_id: work_id,
      token: getToken(),
    }

    async function startFetching() {
      try {
        const json = await fetchCheckMusicAsync(params)
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
  }, [work_id])

  const onSubmit = async () => {
    const list = []
    let canSub = true
    wai: for (let i = 0; i < qusList.length; i++) {
      for (let j = 0; j < qusList[i].ui.length; j++) {
        if (
          qusList[i].ui[j].score === undefined &&
          Object.prototype.hasOwnProperty.call(qusList[i].ui[j], 'source') === true
        ) {
          canSub = false
          alert('有未打分项！')
          break wai
        }
        if (qusList[i].ui[j].score !== undefined) {
          list.push(qusList[i].ui[j].score)
        }
      }
    }
    list.push(0)

    if (canSub) {
      try {
        setLoading(true)
        const res = await pushCheckMusicScoreAsync(work_id, getToken(), list)
        if (Number(res.data.code) === 200) {
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
  }

  const switchComp = (ui_item, ui_index, list_index) => {
    switch (qusList[list_index].ui[ui_index].type) {
      case '0':
        return <EditC eValue={ui_item.desc} disabled={true} pe={'none'} />
      case '10':
        return <TagC tValue={ui_item.tags} disabled={true} pe={'none'} />
      case '11':
        return <TagEditC tValue={ui_item.tags} eValue={ui_item.desc} disabled={true} pe={'none'} />
      case '20':
        return (
          <RadioC
            rValue={ui_item.option_check}
            options={ui_item.options}
            disabled={true}
            pe={'none'}
          />
        )
      case '21':
        return (
          <RadioEditC
            rValue={ui_item.option_check}
            eValue={ui_item.desc}
            options={ui_item.options}
            disabled={true}
            pe={'none'}
          />
        )
      default:
        return null
    }
  }

  const onScoreChange = (q_id, u_id, value) => {
    qusList[q_id].ui[u_id].score = value
    setQusList([...qusList])
    addAllScore()
  }

  const addAllScore = () => {
    let sum_score = 0
    qusList.map((qus) => {
      qus.ui.map((indexUI) => {
        if (indexUI.score === undefined || indexUI.score === '') {
          return 0
        }
        sum_score += indexUI.score
        return indexUI.score
      })
    })
    setAllScore(sum_score)
  }

  return (
    <div>
      {songName !== '' ? <h2>歌名：{songName}</h2> : null}
      {songUrl !== '' ? (
        <div
          style={{ position: 'sticky', top: 0, backgroundColor: 'rgb(232, 237, 241)', zIndex: 1 }}
        >
          <AudioItem src={songUrl} mFirst={true} mToast={2} mPrompt={true} id={songName} />
        </div>
      ) : null}

      <div className="music_form">
        {qusList.map((item, index) => (
          <div key={item.id} className="qus">
            <h3>
              {index + 1}、{item.question}
            </h3>
            {item.ui.map((ui_item, ui_index) => {
              return (
                <div key={`${item.id}_${ui_index}`}>
                  <div
                    style={{ display: 'flex', alignItems: 'center' }}
                    className="qa_ui_item_container"
                  >
                    <h3 className="qa_ui_title">{ui_item.title}：</h3>
                    {Object.prototype.hasOwnProperty.call(ui_item, 'source') ? (
                      <InputNumber
                        min={0}
                        max={ui_item.source}
                        precision={0}
                        className="score_input"
                        size="small"
                        value={ui_item.score}
                        placeholder="请输入分数"
                        onChange={(e) => onScoreChange(index, ui_index, e)}
                      />
                    ) : null}
                  </div>
                  {ui_item.tips !== '' && ui_item.tips !== null && ui_item.tips !== undefined ? (
                    <div className="source_tip">{ui_item.tips}</div>
                  ) : null}
                  {switchComp(ui_item, ui_index, index)}
                </div>
              )
            })}
          </div>
        ))}

        {songUrl !== '' ? (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ fontSize: 'large' }}>
                <strong>总分： {allScore}</strong>
              </div>
              <Button
                type="primary"
                className="button"
                loading={loading}
                onClick={() => onSubmit()}
              >
                提交
              </Button>
            </div>
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
    </div>
  )
}

export default MusicShow