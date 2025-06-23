import { Flex } from 'antd'
import React, { useEffect, useState } from 'react'

import { LyricSelect, useLyricStore } from '../../../../stores/LrcTimestamp'

const LyricwordBox = (props: {
  word: (string | number)[]
  minLeft: number //毫秒
  maxRight: number //毫秒
  multiple: number // 每秒像素
  wordIndex: LyricSelect
}) => {
  const { minLeft, maxRight, multiple, wordIndex } = props
  const startPx = (Number(props.word[0]) / 1000) * multiple // 字 开始位置
  const endPx = (Number(props.word[1]) / 1000) * multiple //  字 结束位置
  const minLeftPx = (minLeft / 1000) * multiple // 字 开始位置
  const maxRightPx = (maxRight / 1000) * multiple //  字 结束位置
  const text = props.word[2].toString()
  const [isHover, setIsHover] = useState(false)

  const selectIndex = useLyricStore((state) => {
    return state.selectIndex
  })
  const selectLyricWord = useLyricStore((state) => {
    return state.selectLyricWord
  })
  const updateSelectWordInfoTime = useLyricStore((state) => {
    return state.updateSelectWordInfoTime
  })

  useEffect(() => {}, [selectIndex])

  const handleResizeLeft = (e) => {
    if (
      selectIndex.lyricIndex !== wordIndex.lyricIndex ||
      selectIndex.wordIndex !== wordIndex.wordIndex
    ) {
      return
    }
    const startX = e.clientX
    document.onmousemove = (e) => {
      const endX = e.clientX
      const changeWidth = endX - startX
      console.log('handleResizeLeftchangeWidth:' + changeWidth)
      let targetLeft = startPx + changeWidth
      if (targetLeft < minLeftPx) {
        targetLeft = minLeftPx
      }
      if (targetLeft >= endPx) {
        // 弹框，是否删除该字
        targetLeft = endPx - 10
      }
      // 将targetLeft， end 像素 转换为毫秒时间戳
      const startMS = Math.ceil((targetLeft / multiple) * 1000)
      const endMS = Math.ceil((endPx / multiple) * 1000)
      updateSelectWordInfoTime(wordIndex, startMS, endMS)
    }
    document.onmouseup = () => {
      document.onmousemove = null
      document.onmouseup = null
    }
  }
  const handleResizeRight = (e) => {
    if (
      selectIndex.lyricIndex !== wordIndex.lyricIndex ||
      selectIndex.wordIndex !== wordIndex.wordIndex
    ) {
      return
    }
    const startX = e.clientX
    document.onmousemove = (e) => {
      const endX = e.clientX
      const changeWidth = endX - startX
      console.log('changeWidth:' + changeWidth)
      let targetRight = endPx + changeWidth
      if (targetRight > maxRightPx) {
        targetRight = maxRightPx
      }
      if (targetRight <= startPx) {
        // 弹框，是否删除该字
        targetRight = startPx + 20
      }
      // 将startPx， targetRight 像素 转换为毫秒时间戳
      const startMS = Math.ceil((startPx / multiple) * 1000)
      const endMS = Math.ceil((targetRight / multiple) * 1000)
      updateSelectWordInfoTime(wordIndex, startMS, endMS)
    }
    document.onmouseup = () => {
      document.onmousemove = null
      document.onmouseup = null
    }
  }

  return (
    <Flex
      id={startPx.toString()}
      justify="center"
      align="center"
      style={{
        width: `${endPx - startPx}px`,
        height: '30px',
        overflow: 'hidden',
        left: `${startPx}px`,
        // cursor: 'pointer',
        position: 'absolute',
        backgroundColor: '#7DE2FC',
        opacity:
          selectIndex.lyricIndex === wordIndex.lyricIndex &&
          selectIndex.wordIndex === wordIndex.wordIndex
            ? 1
            : isHover
              ? 0.9
              : 0.7,
        justifyContent: 'space-between',
      }}
      onClick={() => {
        selectLyricWord(wordIndex)
      }}
      onMouseEnter={() => {
        setIsHover(true)
      }}
      onMouseLeave={() => {
        setIsHover(false)
      }}
    >
      <div
        id={`resizeLeft${startPx}`}
        style={{ float: 'left', width: '1px', height: '100%', cursor: 'w-resize' }}
        onMouseDown={handleResizeLeft}
      ></div>
      {text}
      <div
        id={`resizeRight${endPx}`}
        style={{ float: 'right', width: '1px', height: '100%', cursor: 'e-resize' }}
        onMouseDown={handleResizeRight}
      ></div>
    </Flex>
  )
}

export default LyricwordBox
