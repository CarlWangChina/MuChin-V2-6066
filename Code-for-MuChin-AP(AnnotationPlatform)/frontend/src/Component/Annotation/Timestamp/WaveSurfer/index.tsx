import { Flex, Slider } from 'antd'
import React, { useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import HoverPlugin from 'wavesurfer.js/dist/plugins/hover'
import TimelinePlugin from 'wavesurfer.js/dist/plugins/timeline'
import ZoomPlugin from 'wavesurfer.js/dist/plugins/zoom'

import { useLyricStore } from '../../../../stores/LrcTimestamp'
import { LyricLine } from '../LyricLine'

export function ExampleComponent() {
  const wavesurferRef = useRef<WaveSurfer>()

  const waveformRef = useRef<HTMLDivElement>(null)
  const audioRef = useRef<HTMLAudioElement>()

  const [zoomValue, setZoomValue] = useState(0)
  const [playTime, setPlayTime] = useState(0)

  const formatTimestamp = useLyricStore((state) => {
    return state.formatTimestamp
  })
  const selectLyricWordInfo = useLyricStore((state) => {
    return state.selectLyricWordInfo
  })
  const { selectIndex, clickTime, lyricDuration } = useLyricStore((state) => {
    return {
      selectIndex: state.selectIndex,
      clickTime: state.clickTime,
      lyricDuration: state.lyricDuration,
    }
  })
  const setClickTime = useLyricStore((state) => {
    return state.setClickTime
  })
  const setLyricDuration = useLyricStore((state) => {
    return state.setLyricDuration
  })

  useEffect(() => {}, [selectIndex])

  useEffect(() => {
    if (waveformRef.current) {
      wavesurferRef.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: 'violet',
        progressColor: 'purple',
        autoScroll: true,
        autoCenter: true,
        barGap: 1,
        media: audioRef.current,
        mediaControls: true,
        barWidth: 1,
        plugins: [TimelinePlugin.create(), HoverPlugin.create(), ZoomPlugin.create()],
      })

      return () => {
        wavesurferRef.current?.unAll()
        wavesurferRef.current?.destroy()
      }
    }
  }, [])

  useEffect(() => {
    if (wavesurferRef.current) {
      wavesurferRef.current.on('ready', function (duration) {
        setLyricDuration(Number(duration))
      })
      wavesurferRef.current.on('zoom', function (value) {
        setZoomValue(value)
      })
      wavesurferRef.current.on('audioprocess', function (currentTime) {
        setPlayTime(currentTime)
      })
      wavesurferRef.current.on('interaction', function (newTime) {
        console.log(newTime)
        setClickTime(newTime)
      })
      wavesurferRef.current.on('scroll', function (startTime, endTime) {
        const element = document.getElementById('lyricLine')
        element.scrollLeft = startTime * zoomValue
      })
    }
  }, [zoomValue])

  return (
    <Flex vertical style={{ width: '100%' }}>
      <div>当前播放时间：{formatTimestamp(Math.ceil(playTime * 1000))}</div>
      <div>当前点击时间：{formatTimestamp(Math.ceil(clickTime * 1000))}</div>
      <br />
      <div>
        <p>{selectLyricWordInfo()}</p>
      </div>
      <br />
      <div ref={waveformRef} />
      <LyricLine totalTime={lyricDuration} multiple={zoomValue} currentTime={playTime} />

      <Flex
        justify="center"
        align="center"
        style={{ justifyContent: 'space-between', marginTop: '20px' }}
      >
        <audio ref={audioRef} src="/test.mp3" style={{ width: '80%' }} />
        <Flex justify="center" align="center" style={{ justifyContent: 'flex-end', width: '40%' }}>
          <div>放大倍数：{zoomValue}</div>
          <div style={{ padding: '0px 10px' }}>-</div>
          <Slider
            style={{ width: '30%' }}
            min={5}
            max={200}
            value={zoomValue}
            onChange={(value) => {
              wavesurferRef.current?.zoom(value)
            }}
          />
          <div style={{ padding: '0px 10px' }}>+</div>
        </Flex>
      </Flex>
    </Flex>
  )
}
