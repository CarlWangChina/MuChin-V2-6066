import { Flex } from 'antd'
import React from 'react'

import { useLyricStore } from '../../../../stores/LrcTimestamp'
import LyricwordBox from '../LyricwordBox'

// 歌词操作版
export const LyricLine = (props: { totalTime: number; multiple: number; currentTime: number }) => {
  const { totalTime, multiple } = props

  const lyrics = useLyricStore((state) => {
    return state.lyricTimeStamp
  })
  return (
    <Flex id="lyricLine" style={{ width: '100%', overflow: 'hidden', backgroundColor: '#ff0' }}>
      <Flex
        style={{
          position: 'relative',
          width: `${totalTime * multiple}px`,
          height: '30px',
          alignItems: 'center',
        }}
      >
        {lyrics.map((rowLrc, rowIndex) => {
          const rowMinleft = rowIndex > 0 ? lyrics[rowIndex - 1].end : 0
          const rowMaxright =
            rowIndex + 1 >= lyrics.length ? totalTime * 1000 : lyrics[rowIndex + 1].start
          return rowLrc.words.map((wordInfo, wordIndex) => {
            const minLeft = wordIndex > 0 ? Number(rowLrc.words[wordIndex - 1][1]) : rowMinleft
            const maxRight =
              wordIndex + 1 >= rowLrc.words.length
                ? rowMaxright
                : Number(rowLrc.words[wordIndex + 1][0])
            return (
              <LyricwordBox
                key={wordInfo[1]}
                word={wordInfo}
                minLeft={minLeft}
                maxRight={maxRight}
                multiple={multiple}
                wordIndex={{ lyricIndex: rowIndex, wordIndex: wordIndex }}
              />
            )
          })
        })}
      </Flex>
    </Flex>
  )
}
