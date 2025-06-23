import './index.scss'

import { Modal } from 'antd'
import { useEffect } from 'react'

import { getAlert, getFirst, setAlert, setFirst } from '../../../utils/auth'

const Player = ({ mUrl, mToast, mPrompt }) => {

  let modal

  useEffect(() => {
    window.onhashchange = function () {
      if (modal !== undefined) {
        try {
          modal.destroy()
        } catch (err) {
          console.error(err)
        }
      }
    }
  }, [modal])

  const canPlay = () => {

    var audio = document.getElementById(`myAudio`)

    if (getAlert() === 'true') {
      modal = Modal.info({
        title: '提示',
        content: (
          <div>
            <p>由于浏览器内核适配问题，刷新后音乐可能无法自动播放。</p>
          </div>
        ),
        onOk() {
          setAlert(false)
          if (mToast === 2) {
            audio.play()
          }
        },
      })

    }
  }

  const endAudio = () => {
    if (mToast === 2 && getFirst() === 'true') {
      setFirst(false)
    }
  }

  return (
    <div className="player">
      <audio
        id="myAudio"
        className="audio_player"
        src={mUrl}
        style={{ width: '100%', pointerEvents: 'auto' }}
        controls="controls"
        autoPlay
        loop
        controlsList={mPrompt === true ? 'nodownload noplaybackrate ' : 'nodownload'}
        onCanPlay={() => canPlay()}
        onEnded={() => endAudio()}
      />
      {mToast === 2 ? null : (
        <div style={{ color: 'red' }} className="player_tip">
          如果未自动播放，请手动点击一下播放按钮
        </div>
      )}
    </div>
  )
}
export default Player