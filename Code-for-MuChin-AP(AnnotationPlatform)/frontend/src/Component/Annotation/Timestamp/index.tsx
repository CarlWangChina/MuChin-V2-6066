import { Button, Flex, Input, Modal, Select, Space } from 'antd'
import React, { useEffect, useState } from 'react'

import { fetchLyricsAsync2 } from '../../../api/lyrics'
import { useLyricStore } from '../../../stores/LrcTimestamp'
import { getToken } from '../../../utils/auth'
import LyricOperation from './LyricOperation'
import { ExampleComponent } from './WaveSurfer'

const TimestampPage = (props) => {
  const formatTimestamp = useLyricStore((state) => {
    return state.formatTimestamp
  })
  const updateTotalTimestamp = useLyricStore((state) => {
    return state.updateTotalTimestamp
  })
  const { mid, type } = props

  const [songName, setSongName] = useState('')
  const [songUrl, setSongUrl] = useState('')
  const [opts, setOpts] = useState([])

  useEffect(() => {
    let ignore = false

    const params = {
      song_id: mid,
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
            // const list = setInfoData(json.data.data.info)
            // setLyric(list)
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
  }, [mid])

  return (
    <Flex vertical style={{ justifyContent: 'center' }}>
      <h1>{songName}</h1>
      <br />
      <MoveLyricTotal
        changeTotal={(value) => {
          updateTotalTimestamp(value)
        }}
      />
      <br />
      <ExampleComponent />
      <br />
      <LyricOperation />
    </Flex>
  )
}
// 全量移动
const MoveLyricTotal = (props: { changeTotal: (value: number) => void }) => {
  const { changeTotal } = props
  const [inputValue, setInputValue] = useState('')
  const [openModal, setOpenModal] = useState(false)
  const [skipInfo, setSkipInfo] = useState('向前')

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { value } = e.target
    const reg = /^\d+$/
    if (reg.test(value) || value === '') {
      setInputValue(value)
    }
  }

  const handleOk = () => {
    let microsecond = parseInt(inputValue)
    if (skipInfo === '向前') {
      microsecond = -microsecond
    }
    changeTotal(microsecond)
    setOpenModal(false)
  }

  const handleCancel = () => {
    setOpenModal(false)
  }
  return (
    <Flex vertical>
      <p>所有歌词：</p>
      <Flex>
        <Space.Compact style={{ width: '100%' }}>
          <Select
            value={skipInfo}
            options={[
              {
                value: '向前',
                label: '向前',
              },
              {
                value: '向后',
                label: '向后',
              },
            ]}
            onChange={(value) => {
              setSkipInfo(value)
            }}
          />
          <Input
            value={inputValue}
            style={{ width: '180px' }}
            placeholder="毫秒"
            maxLength={5}
            onChange={handleChange}
          />
          <Button
            type="primary"
            onClick={() => {
              setOpenModal(true)
            }}
          >
            移动
          </Button>
        </Space.Compact>
      </Flex>
      <Modal title="是否修改" open={openModal} onOk={handleOk} onCancel={handleCancel}>
        <p>
          是否将所有歌词时间刻度 {skipInfo} 移动 {inputValue} 毫秒？
        </p>
      </Modal>
    </Flex>
  )
}

export default TimestampPage
