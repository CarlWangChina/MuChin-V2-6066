import { Button } from 'antd'
import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'

import { fetchCheckMusicAsyncSpecail } from '../../../api/check'
import { getFirst, setAlert } from '../../../utils/auth'
import EditC from '../../Annotation/Step1/EditC'
import RadioC from '../../Annotation/Step1/RadioC'
import RadioEditC from '../../Annotation/Step1/RadioEditC'
import TagC from '../../Annotation/Step1/TagC'
import TagEditC from '../../Annotation/Step1/TagEditC'
import AudioItem from '../../Base/Audio'
import withRouter from '../../Base/WithRouter/withRouter'

const CheckMusicShow = (props) => {
  const { work_id, userId, token } = props.params
  const location = useLocation()
  const queryParams = new URLSearchParams(location.search)
  const searchValue = queryParams.get('searchValue')

  const initCanWrite = !(getFirst() === 'true')
  const [canWrite, setCanWrite] = useState(initCanWrite)

  const navigate = useNavigate()

  const [songName, setSongName] = useState('')
  const [songUrl, setSongUrl] = useState('')

  const [qusList, setQusList] = useState([])

  const [pending, setPending] = useState('')

  const [gradleTotal, setGradleTotal] = useState(0)

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
      uid: userId,
      work_id: work_id,
      token: token,
    }

    async function startFetching() {
      try {
        const json = await fetchCheckMusicAsyncSpecail(params)
        if (!ignore) {
          if (Number(json.data.code) === 200) {
            const data = json.data.data
            setSongUrl(data.song_url)
            setSongName(data.song_name)
            if (Object.prototype.hasOwnProperty.call(data, 'pending')) {
              setPending(data.pending)
            } else {
              setQusList(data.qa)
              setGradleTotal(data.gradle_total)
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
  }, [])

  const switchComp = (ui_item, ui_index, list_index) => {
    switch (qusList[list_index].ui[ui_index].type) {
      case '0':
        return <EditC eValue={ui_item.desc} disabled={true} pe={'none'} />
      case '10':
        return <TagC tValue={ui_item.tags} disabled={true} pe={'none'} />
      case '11':
        return <TagEditC tValue={ui_item.tags} disabled={true} eValue={ui_item.desc} pe={'none'} />
      case '20':
        return (
          <RadioC
            rValue={ui_item.option_check}
            disabled={true}
            options={ui_item.options}
            pe={'none'}
          />
        )
      case '21':
        return (
          <RadioEditC
            rValue={ui_item.option_check}
            disabled={true}
            eValue={ui_item.desc}
            options={ui_item.options}
            pe={'none'}
          />
        )
      default:
        return null
    }
  }
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
          <AudioItem src={songUrl} mFirst={true} mToast={2} mPrompt={true} id={songName} />
        </div>
      ) : null}
      {pending !== '' ? (
        <div style={{ fontSize: 'large' }}>
          <strong>该歌曲被挂起</strong>
          <div>原因如下：</div>
          <div>{pending}</div>
        </div>
      ) : (
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
                      {Object.prototype.hasOwnProperty.call(ui_item, 'gradle') ? (
                        <h3>{ui_item.gradle}分</h3>
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

          {gradleTotal !== undefined ? (
            <div style={{ paddingBottom: 50 }}>
              <div
                style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
              >
                <div style={{ fontSize: 'large' }}>
                  <strong>总分： {gradleTotal}分</strong>
                </div>
              </div>

            </div>
          ) : null}
        </div>
      )}
      {
        <Button type="primary" className="check_show_btn" onClick={onNavList}>
          回到当前列表
        </Button>
      }
    </div>
  )
}

export default withRouter(CheckMusicShow)