import { produce } from 'immer'
import { create } from 'zustand'

import timestamp from '../../Component/Annotation/Timestamp/music_time.json'

export interface LyricModel {
  start: number
  end: number
  text: string
  words: (string | number)[][]
}
export interface LyricSelect {
  lyricIndex: number
  wordIndex: number
}

interface LrcTimestampInterface {
  // 当前歌词 时间戳信息
  lyricTimeStamp: Array<LyricModel>

  // 当前选中 歌词行+字 【index index】
  // 当前未选中【-1， -1】
  // 选中index行，未选中字【index, -1】
  selectIndex: LyricSelect

  selectLyricWord: (index: LyricSelect) => void
  selectLyricWordInfo: () => string

  clickTime: number // 当前hover点击 毫秒级时间戳
  lyricDuration: number // 歌曲总时间
  // 修改当前选中 歌词字 时间戳 start end word
  updateSelectWordInfoTime: (index: LyricSelect, start: number, end: number) => void
  updateSelectWordInfoTimeChecked: (
    index: LyricSelect,
    start: number,
    end: number,
    word?: string
  ) => { successed: boolean; info: string }

  updateTotalTimestamp: (microsecond: number) => void

  formatTimestamp: (timestamp: number) => string

  setClickTime: (time: number) => void
  setLyricDuration: (duration: number) => void
  // 删除歌词字
  deleteLyricWord: (index: LyricSelect) => void
  // 删除歌词行
  deleteLyricRow: (index: LyricSelect) => void
  // 插入歌词行
  insertLyricRow: (
    rowText: string,
    start: number,
    end: number,
    wordInfo: (string | number)[][]
  ) => boolean
  // 插入歌词字
  insertLyricWord: (
    select: LyricSelect,
    position: string,
    start: number,
    end: number,
    word: string
  ) => { successed: boolean; info: string }
}
export const useLyricStore = create<LrcTimestampInterface>()((set, get) => {
  const jsonStr = timestamp.timestamp.toString()
  return {
    lyricTimeStamp: timestamp.timestamp,
    selectIndex: { lyricIndex: -1, wordIndex: -1 },
    clickTime: 0,
    lyricDuration: 0,
    selectLyricWord(index) {
      set((state) => {
        return produce(state, (draft) => {
          draft.selectIndex = { ...index }
        })
      })
    },
    updateSelectWordInfoTime(index, start, end) {
      const selectIndex = get().selectIndex
      if (
        selectIndex.lyricIndex !== index.lyricIndex ||
        selectIndex.wordIndex !== index.wordIndex
      ) {
        return
      }
      if (selectIndex.lyricIndex === -1) {
        return
      }
      if (selectIndex.wordIndex === -1) {
        return
      }
      const words = get().lyricTimeStamp[selectIndex.lyricIndex].words[selectIndex.wordIndex]
      const newWords = [start, end, words[2]]
      let rowStart = get().lyricTimeStamp[selectIndex.lyricIndex].start
      let rowEnd = get().lyricTimeStamp[selectIndex.lyricIndex].end

      if (selectIndex.wordIndex === 0) {
        rowStart = start
      }
      if (selectIndex.wordIndex === get().lyricTimeStamp[selectIndex.lyricIndex].words.length - 1) {
        rowEnd = end
      }

      set((state) => {
        return produce(state, (draft) => {
          draft.lyricTimeStamp[selectIndex.lyricIndex].start = rowStart
          draft.lyricTimeStamp[selectIndex.lyricIndex].end = rowEnd
          draft.lyricTimeStamp[selectIndex.lyricIndex].words[selectIndex.wordIndex] = newWords
        })
      })
    },
    updateSelectWordInfoTimeChecked(index, start, end, word) {
      const selectIndex = get().selectIndex
      if (
        selectIndex.lyricIndex !== index.lyricIndex ||
        selectIndex.wordIndex !== index.wordIndex
      ) {
        return
      }
      if (selectIndex.lyricIndex === -1) {
        return
      }
      if (selectIndex.wordIndex === -1) {
        return
      }
      const words = get().lyricTimeStamp[index.lyricIndex].words[index.wordIndex]
      let minStart = 0
      let maxEnd = get().lyricDuration * 1000

      // 校验起始 结束位置
      if (start >= end) {
        return { successed: false, info: '结束时间不能小于起始时间' }
      }

      // 校验start  index=0 ,start< 上一行 最后字的 end;
      if (index.wordIndex === 0) {
        minStart = index.lyricIndex > 0 ? get().lyricTimeStamp[index.lyricIndex - 1].end : 0
      } else {
        const lastWord = get().lyricTimeStamp[index.lyricIndex].words[index.wordIndex - 1]
        minStart = Number(lastWord[1])
      }
      if (start < minStart) {
        return { successed: false, info: '起始时间不能小于上一个字的结束时间' }
      }

      // 校验end
      if (index.wordIndex === get().lyricTimeStamp[index.lyricIndex].words.length - 1) {
        maxEnd =
          index.lyricIndex === get().lyricTimeStamp.length - 1
            ? get().lyricDuration * 1000
            : get().lyricTimeStamp[index.lyricIndex + 1].start
      } else {
        const nextWord = get().lyricTimeStamp[index.lyricIndex].words[index.wordIndex + 1]
        maxEnd = Number(nextWord[0])
      }
      if (end > maxEnd) {
        return { successed: false, info: '结束时间不能大于下一个字的起始时间' }
      }

      const editWord = word === null ? words[2] : word
      const newWords = [start, end, editWord]
      let rowStart = get().lyricTimeStamp[selectIndex.lyricIndex].start
      let rowEnd = get().lyricTimeStamp[selectIndex.lyricIndex].end

      if (selectIndex.wordIndex === 0) {
        rowStart = start
      }
      if (selectIndex.wordIndex === get().lyricTimeStamp[selectIndex.lyricIndex].words.length - 1) {
        rowEnd = end
      }

      set((state) => {
        return produce(state, (draft) => {
          draft.lyricTimeStamp[selectIndex.lyricIndex].start = rowStart
          draft.lyricTimeStamp[selectIndex.lyricIndex].end = rowEnd
          draft.lyricTimeStamp[selectIndex.lyricIndex].words[selectIndex.wordIndex] = newWords
          draft.lyricTimeStamp[selectIndex.lyricIndex].text = draft.lyricTimeStamp[
            selectIndex.lyricIndex
          ].words
            .map((v) => {
              return v[2]
            })
            .join('')
        })
      })
      return { successed: true, info: '修改成功' }
    },
    formatTimestamp(timestamp) {
      const millisecond = timestamp % 1000
      const second = ((timestamp - millisecond) / 1000) % 60
      const minute = ((timestamp - millisecond) / 1000 - second) / 60
      const timeStr = `${minute < 10 ? '0' : ''}${minute}:${second < 10 ? '0' : ''}${second}.${millisecond}`
      return timeStr.padEnd(9, '0')
    },
    updateTotalTimestamp(microsecond) {
      const lyricTimeStamp = get().lyricTimeStamp
      const lyricTimeStampEdit = lyricTimeStamp.map((row) => {
        const wordsEdit = row.words.map((word) => {
          return [Number(word[0]) + microsecond, Number(word[1]) + microsecond, word[2]]
        })
        return {
          ...row,
          start: row.start + microsecond,
          end: row.end + microsecond,
          words: wordsEdit,
        }
      })
      console.log(microsecond)
      console.log(lyricTimeStamp)
      set((state) => {
        return produce(state, (draft) => {
          draft.lyricTimeStamp = [...lyricTimeStampEdit]
        })
      })
    },
    selectLyricWordInfo() {
      const selectIndex = get().selectIndex
      if (selectIndex.lyricIndex < 0 || selectIndex.wordIndex < 0) {
        return '未选中'
      }
      const text = get().lyricTimeStamp[selectIndex.lyricIndex].text
      const words = get().lyricTimeStamp[selectIndex.lyricIndex].words[selectIndex.wordIndex]
      const formatTimestamp = get().formatTimestamp
      return `${text}\n${words[2]}:\n${formatTimestamp(Number(words[0].toString()))}\n${formatTimestamp(Number(words[1].toString()))}`
    },
    setClickTime(time) {
      set((state) => {
        return produce(state, (draft) => {
          draft.clickTime = time
        })
      })
    },
    setLyricDuration(duration) {
      set((state) => {
        return produce(state, (draft) => {
          draft.lyricDuration = duration
        })
      })
    },
    deleteLyricWord(index) {
      if (index.lyricIndex < 0 || index.wordIndex < 0) {
        return
      }
      set((state) => {
        return produce(state, (draft) => {
          draft.selectIndex = { lyricIndex: -1, wordIndex: -1 }
          draft.lyricTimeStamp[index.lyricIndex].words.splice(index.wordIndex, 1)
          draft.lyricTimeStamp[index.lyricIndex].text = draft.lyricTimeStamp[index.lyricIndex].words
            .map((v) => {
              return v[2]
            })
            .join('')
        })
      })
    },
    deleteLyricRow(index) {
      if (index.lyricIndex < 0 || index.wordIndex < 0) {
        return
      }
      set((state) => {
        return produce(state, (draft) => {
          draft.selectIndex = { lyricIndex: -1, wordIndex: -1 }
          draft.lyricTimeStamp.splice(index.lyricIndex, 1)
        })
      })
    },
    insertLyricRow(rowText, start, end, wordInfo) {
      const row: LyricModel = {
        start: start,
        end: end,
        text: rowText,
        words: wordInfo,
      }
      let insertIndex = -1
      get().lyricTimeStamp.forEach((row, index, array) => {
        if (index === 0 && end <= row.start) {
          console.log(`end:${end}; row.start${row.start}`)
          insertIndex = 0
        } else if (index === array.length - 1 && start >= row.end && end <= get().lyricDuration) {
          insertIndex = array.length - 1
        } else if (start >= row.end && end <= array[index + 1].start) {
          insertIndex = index + 1
        }
      })

      if (insertIndex === -1) {
        return false
      }
      set((state) => {
        return produce(state, (draft) => {
          draft.lyricTimeStamp.splice(insertIndex, 0, row)
        })
      })
      return true
    },
    insertLyricWord(select, position, start, end, word) {
      if (select.lyricIndex < 0 || select.wordIndex < 0) {
        return
      }
      if (start >= end) {
        return { successed: false, info: '结束时间不能小于起始时间' }
      }
      let insertIndex = 0
      let minStart = 0
      let maxEnd = 0
      const currentWord = get().lyricTimeStamp[select.lyricIndex].words[select.wordIndex]
      console.log(position)
      if (position === 'before') {
        insertIndex = select.wordIndex
        // 选中字之前插入；start end
        if (select.wordIndex === 0) {
          minStart = select.lyricIndex > 0 ? get().lyricTimeStamp[select.lyricIndex - 1].end : 0
        } else {
          const lastWord = get().lyricTimeStamp[select.lyricIndex].words[select.wordIndex - 1]
          minStart = Number(lastWord[1])
        }
        if (start < minStart) {
          return { successed: false, info: '起始时间不能小于上一个字的结束时间' }
        }
        maxEnd = Number(currentWord[0])
        if (end > maxEnd) {
          return { successed: false, info: '结束时间不能大于当前选中字的起始时间' }
        }
      } else {
        insertIndex = select.wordIndex + 1
        // 选中字之后插入；start end
        minStart = Number(currentWord[1])
        if (start < minStart) {
          return { successed: false, info: '起始时间不能小于当前选中字的结束时间' }
        }

        // 校验end
        if (select.wordIndex === get().lyricTimeStamp[select.lyricIndex].words.length - 1) {
          maxEnd =
            select.lyricIndex === get().lyricTimeStamp.length - 1
              ? get().lyricDuration * 1000
              : get().lyricTimeStamp[select.lyricIndex + 1].start
        } else {
          const nextWord = get().lyricTimeStamp[select.lyricIndex].words[select.wordIndex + 1]
          maxEnd = Number(nextWord[0])
        }
        if (end > maxEnd) {
          return { successed: false, info: '结束时间不能大于下一个字的起始时间' }
        }
      }
      const newWords = [start, end, word]
      let rowStart = get().lyricTimeStamp[select.lyricIndex].start
      let rowEnd = get().lyricTimeStamp[select.lyricIndex].end

      if (insertIndex === 0) {
        rowStart = start
      }
      if (insertIndex >= get().lyricTimeStamp[select.lyricIndex].words.length - 1) {
        rowEnd = end
      }
      console.log(newWords)

      set((state) => {
        return produce(state, (draft) => {
          draft.lyricTimeStamp[select.lyricIndex].start = rowStart
          draft.lyricTimeStamp[select.lyricIndex].end = rowEnd
          draft.lyricTimeStamp[select.lyricIndex].words.splice(insertIndex, 0, newWords)
          draft.lyricTimeStamp[select.lyricIndex].text = draft.lyricTimeStamp[
            select.lyricIndex
          ].words
            .map((v) => {
              return v[2]
            })
            .join('')
        })
      })
      return { successed: true, info: '插入成功' }
    },
  }
})