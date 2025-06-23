import './index.scss'

import { CaretRightOutlined, MoreOutlined, PauseOutlined, SoundFilled } from '@ant-design/icons'
import { Dropdown } from 'antd'
import { Component } from 'react'

import { getFirst, setFirst } from '../../../utils/auth'

class App extends Component {
  constructor(props) {
    super(props)
    this.state = {
      rateList: [0.5, 1.0, 1.25, 1.5, 2.0],
      playRate: 1.0,
      isPlay: false,
      isMuted: false,
      volume: 100,
      allTime: 0,
      currentTime: 0,
      showVolume: false,
      showRate: false,
      sliderDisable: getFirst() === 'true',
    }
  }

  componentDidMount() {}

  formatSecond(time) {
    const microsecond_str = time.toFixed(2).split('.')[1]

    const second = Math.floor(time % 60)
    const second_str = second >= 10 ? second : `0${second}`

    const minite = Math.floor(time / 60)
    const minite_str = minite >= 10 ? minite : `0${minite}`

    return minite_str + ':' + second_str + '.' + microsecond_str
  }

  onCanPlay = () => {
    const { id } = this.props
    const audio = document.getElementById(`audio${id}`)
    this.setState({
      allTime: audio.duration,
    })
  }

  onPlay = () => {
    this.setState({
      isPlay: true,
    })
    if (getFirst() === 'true') {
      this.setState({
        sliderDisable: true,
      })
    }
  }
  onEnded = () => {
    const { id } = this.props
    const audio = document.getElementById(`audio${id}`)
    audio.play()
    setFirst(false)
    this.setState({
      sliderDisable: false,
    })
  }

  playAudio = () => {
    const { id } = this.props
    const audio = document.getElementById(`audio${id}`)
    audio.play()
    this.setState({
      isPlay: true,
    })
  }

  pauseAudio = () => {
    const { id } = this.props
    const audio = document.getElementById(`audio${id}`)
    audio.pause()
    this.setState({
      isPlay: false,
    })
  }

  onMuteAudio = () => {
    const { id } = this.props
    const audio = document.getElementById(`audio${id}`)
    this.setState({
      isMuted: !audio.muted,
    })
    audio.muted = !audio.muted
  }

  changeTime = (e) => {
    const { value } = e.target
    const { id } = this.props
    if (getFirst() === 'true') {
      console.log('第一次没播放完毕')
      return
    }
    const audio = document.getElementById(`audio${id}`)
    this.setState({
      currentTime: value,
    })
    audio.currentTime = value

    if (value === audio.duration) {
      this.setState({
        isPlay: false,
        sliderDisable: false,
      })
    }
  }

  onTimeUpdate = () => {
    const { id } = this.props
    const audio = document.getElementById(`audio${id}`)

    this.setState({
      currentTime: audio.currentTime,
    })
    if (audio.currentTime === audio.duration) {
      this.setState({
        isPlay: false,
        sliderDisable: false,
      })
    }
  }

  changeVolume = (e) => {
    const { value } = e.target
    const { id } = this.props
    const audio = document.getElementById(`audio${id}`)
    audio.volume = value / 100

    this.setState({
      volume: value,
      isMuted: !value,
    })
  }

  showingVolume = () => {
    this.setState({
      showVolume: true,
    })
  }
  hiddenVolume = () => {
    this.setState({
      showVolume: false,
    })
  }
  showingRate = () => {
    this.setState({
      showRate: true,
    })
  }
  hiddenRate = () => {
    this.setState({
      showRate: false,
    })
  }

  changePlayRate = (num) => {
    this.audioDom.playbackRate = num
    this.setState({
      playRate: num,
    })
  }

  render() {
    const { src, id, mToast } = this.props

    const {
      isPlay,
      isMuted,
      volume,
      allTime,
      currentTime,
      rateList,
      playRate,
      showVolume,
      showRate,
      sliderDisable,
    } = this.state
    const items = [
      {
        key: '0.5',
        label: '0.5倍',
      },
      {
        key: '1',
        label: '1倍',
      },
      {
        key: '1.25',
        label: '1.25倍',
      },
      {
        key: '1.5',
        label: '1.5倍',
      },
      {
        key: '2.0',
        label: '2.0倍',
      },
    ]
    const onClick = ({ key }) => {
      this.changePlayRate(key)
    }

    return (
      <div>
        <div className="audio_content">
          <audio
            id={`audio${id}`}
            src={src}
            autoPlay
            ref={(audio) => {
              this.audioDom = audio
            }}
            preload={'auto'}
            onCanPlay={this.onCanPlay}
            onTimeUpdate={this.onTimeUpdate}
            onPlay={this.onPlay}
            onEnded={this.onEnded}
          >
            <track src={src} kind="captions" />
          </audio>

          <div className="play" onClick={isPlay ? this.pauseAudio : this.playAudio}>
            {isPlay ? (
              <PauseOutlined style={{ fontSize: '18px' }} />
            ) : (
              <CaretRightOutlined style={{ fontSize: '18px' }} />
            )}
          </div>

          <div className="slider_content">
            <span style={{ fontSize: '14px', marginRight: '10px' }}>
              {this.formatSecond(currentTime) + ' / ' + this.formatSecond(allTime)}
            </span>
            <input
              className="slider"
              type="range"
              step="0.01"
              max={allTime}
              value={currentTime}
              onChange={this.changeTime}
              disabled={false}
            />
          </div>

          <div
            className="volume"
            onMouseEnter={this.showingVolume}
            onMouseLeave={this.hiddenVolume}
          >
            {showVolume && (
              <input
                style={{ marginRight: '10px' }}
                type="range"
                onChange={this.changeVolume}
                value={isMuted ? 0 : volume}
              />
            )}

            <SoundFilled style={{ fontSize: '18px' }} onClick={this.onMuteAudio} />
          </div>

          {sliderDisable ? null : (
            <div className="playRate">
              {showRate &&
                rateList &&
                rateList.length > 0 &&
                rateList.map((item) => (
                  <button
                    key={item}
                    style={
                      playRate === item
                        ? {
                            border: '1px solid
                            color: '
                          }
                        : null
                    }
                    onClick={() => this.changePlayRate(item)}
                  >
                    {item}
                  </button>
                ))}
              <Dropdown
                arrow={true}
                value={{ playRate }}
                menu={{ items, onClick }}
                title="播放速率"
                placement="bottomLeft"
              >
                <MoreOutlined style={{ fontSize: '18px' }} />
              </Dropdown>
            </div>
          )}
        </div>
        {mToast === 2 ? null : (
          <div style={{ color: 'red' }} className="player_tip">
            如果未自动播放，请手动点击一下播放按钮
          </div>
        )}
      </div>
    )
  }
}

export default App