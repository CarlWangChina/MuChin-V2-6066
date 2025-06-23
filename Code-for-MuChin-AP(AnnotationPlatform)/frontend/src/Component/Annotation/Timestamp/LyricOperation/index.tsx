import {
  CheckCircleOutlined,
  DeleteOutlined,
  MinusCircleOutlined,
  PlusOutlined,
} from '@ant-design/icons'
import { Button, Flex, Form, Input, message, Popconfirm, Radio, Select, Space } from 'antd'
import React, { useEffect, useState } from 'react'

import { useLyricStore } from '../../../../stores/LrcTimestamp'
import InsertLyric, { SubmitButton } from '../Drawer/InsertLyric/Index'

const LyricOperation = () => {
  const [form] = Form.useForm()
  const [wordChangeable, setWordChangeable] = useState(false)
  const [insertable, setInsertable] = useState(false)
  const values = Form.useWatch([], form)
  const [messageApi, contextHolder] = message.useMessage()
  const changePath = ['start', 'word', 'end']
  const [insertPosition, setInsertPosition] = useState('before')
  const { lyricTimeStamp, selectIndex } = useLyricStore((state) => {
    return {
      lyricTimeStamp: state.lyricTimeStamp,
      selectIndex: state.selectIndex,
      clickTime: state.clickTime,
    }
  })
  const formatTimestamp = useLyricStore((state) => {
    return state.formatTimestamp
  })
  const deleteLyricWord = useLyricStore((state) => {
    return state.deleteLyricWord
  })
  const deleteLyricRow = useLyricStore((state) => {
    return state.deleteLyricRow
  })
  const updateSelectWordInfoTimeChecked = useLyricStore((state) => {
    return state.updateSelectWordInfoTimeChecked
  })
  const insertLyricWord = useLyricStore((state) => {
    return state.insertLyricWord
  })

  function changeLyricWord() {
    const start = form.getFieldValue('start')
    const end = form.getFieldValue('end')
    const word = form.getFieldValue('word')

    const resData = updateSelectWordInfoTimeChecked(
      selectIndex,
      stringTimeToSeconds(start) * 1000,
      stringTimeToSeconds(end) * 1000,
      word
    )
    if (resData.successed) {
      success(resData.info)
    } else {
      error(resData.info)
    }
  }
  function addLyricWord() {
    const insert = form.getFieldValue('word_insert_position').map((word) => {
      const start = stringTimeToSeconds(word['start']) * 1000
      const end = stringTimeToSeconds(word['end']) * 1000
      const wordText: string = word['word']
      console.log(word)
      return { start, end, wordText }
    })
    const { start, end, wordText } = insert[0]
    const resData = insertLyricWord(selectIndex, insertPosition, start, end, wordText)
    if (resData.successed) {
      success(resData.info)
    } else {
      error(resData.info)
    }
  }
  function stringTimeToSeconds(time: string) {
    const time1Value = time.split(':').reduce((minutes, seconds) => {
      return (Number(minutes) * 60 + Number(seconds)).toString()
    })
    return Number(time1Value)
  }
  useEffect(() => {
    console.log(
      `selectIndex.lyricIndex ${selectIndex.lyricIndex},wordIndex:${selectIndex.wordIndex}`
    )
    if (selectIndex.lyricIndex < 0 || selectIndex.wordIndex < 0) {
      form.resetFields()
    } else {
      form.setFieldsValue({
        word: lyricTimeStamp[selectIndex.lyricIndex].words[selectIndex.wordIndex][2].toString(),
        start: formatTimestamp(
          Number(lyricTimeStamp[selectIndex.lyricIndex].words[selectIndex.wordIndex][0].toString())
        ),
        end: formatTimestamp(
          Number(lyricTimeStamp[selectIndex.lyricIndex].words[selectIndex.wordIndex][1].toString())
        ),
      })
    }
  }, [selectIndex, lyricTimeStamp])
  React.useEffect(() => {
    form
      .validateFields({
        validateOnly: true,
      })
      .then(() => {
        setWordChangeable(true)
        setInsertable(true)
      })
      .catch((errorInfo) => {
        const errorFields = errorInfo['errorFields'].map((errorField) => {
          return errorField['name']
        })
        console.log(errorFields)
        let insert = true
        const a = errorFields.filter((v) => {
          if (v[0] === 'word_insert_position') {
            insert = false
          }
          return changePath.indexOf(v[0]) > -1
        })
        // console.log(a)
        setWordChangeable(a.length <= 0)
        setInsertable(insert)
      })
  }, [form, values])
  React.useEffect(() => {}, [insertable, wordChangeable])
  const success = (info) => {
    messageApi.open({
      type: 'success',
      content: info,
    })
  }

  const error = (info) => {
    messageApi.open({
      type: 'error',
      content: info,
    })
  }
  return (
    <>
      {contextHolder}
      <Flex vertical gap="middle">
        {selectIndex.lyricIndex > -1 && selectIndex.wordIndex > -1 ? (
          <Form
            form={form}
            name="lyric-update"
            autoComplete="off"
            variant="filled"
            style={{ maxWidth: 800 }}
          >
            <Form.Item label="选中">
              <Space>
                <Form.Item
                  name="start"
                  style={{ marginBottom: 0 }}
                  rules={[
                    { required: true, message: '请输入开始时间。' },
                    {
                      pattern: new RegExp(/^(\d{2}):([0-5][0-9]).(\d{3})$/),
                      message: '格式xx:xx.xxx',
                    },
                  ]}
                >
                  <Input placeholder="请输入开始时间" />
                </Form.Item>
                <Form.Item
                  name="word"
                  style={{ marginBottom: 0 }}
                  rules={[{ required: true, message: '输入一个字' }]}
                >
                  <Input maxLength={1} style={{ textAlign: 'center' }} placeholder="歌词字" />
                </Form.Item>
                <Form.Item
                  name="end"
                  style={{ marginBottom: 0 }}
                  rules={[
                    { required: true, message: '请输入结束时间。' },
                    {
                      pattern: new RegExp(/^(\d{2}):([0-5][0-9]).(\d{3})$/),
                      message: '格式xx:xx.xxx',
                    },
                  ]}
                >
                  <Input placeholder="请输入结束时间" />
                </Form.Item>
                <Button
                  type="primary"
                  disabled={!wordChangeable}
                  onClick={() => {
                    changeLyricWord()
                  }}
                >
                  修改
                </Button>
                <Popconfirm
                  placement="rightTop"
                  title="确认删除该字吗?"
                  description="确认删除"
                  okText="删除"
                  cancelText="取消"
                  onConfirm={() => {
                    deleteLyricWord(selectIndex)
                  }}
                >
                  <Button danger icon={<DeleteOutlined />}>
                    删除该字
                  </Button>
                </Popconfirm>
              </Space>
            </Form.Item>
            <Form.List name="word_insert_position">
              {(fields, { add, remove }) => (
                <>
                  <Space>
                    <Form.Item
                      // name="position"
                      label={`在所选字【${lyricTimeStamp[selectIndex.lyricIndex].words[selectIndex.wordIndex][2].toString()}】`}
                      rules={[{ required: true, message: '选择插入位置' }]}
                    >
                      <Select
                        placeholder="位置"
                        value={insertPosition}
                        options={[
                          {
                            value: 'before',
                            label: '之前',
                          },
                          {
                            value: 'after',
                            label: '之后',
                          },
                        ]}
                        onChange={(value) => {
                          setInsertPosition(value)
                        }}
                      />
                    </Form.Item>

                    {fields.length <= 0 ? (
                      <Form.Item>
                        <Button type="primary" onClick={() => add()} block icon={<PlusOutlined />}>
                          插入字
                        </Button>
                      </Form.Item>
                    ) : null}
                  </Space>
                  {fields.map(({ key, name, ...restField }) => (
                    <Space key={key} style={{ display: 'flex', marginBottom: 8 }} align="baseline">
                      <Form.Item
                        {...restField}
                        name={[name, 'start']}
                        rules={[
                          { required: true, message: '请输入开始时间。' },
                          {
                            pattern: new RegExp(/^(\d{2}):([0-5][0-9]).(\d{3})$/),
                            message: '格式xx:xx.xxx',
                          },
                        ]}
                      >
                        <Input placeholder="请输入开始时间" />
                      </Form.Item>
                      <Form.Item
                        {...restField}
                        name={[name, 'word']}
                        rules={[{ required: true, message: '输入一个字' }]}
                      >
                        <Input maxLength={1} style={{ textAlign: 'center' }} placeholder="歌词字" />
                      </Form.Item>
                      <Form.Item
                        {...restField}
                        name={[name, 'end']}
                        rules={[
                          { required: true, message: '请输入结束时间。' },
                          {
                            pattern: new RegExp(/^(\d{2}):([0-5][0-9]).(\d{3})$/),
                            message: '格式xx:xx.xxx',
                          },
                        ]}
                      >
                        <Input placeholder="请输入结束时间" />
                      </Form.Item>
                      <Button
                        type="text"
                        onClick={() => remove(name)}
                        style={{ padding: '3px 6px' }}
                      >
                        <MinusCircleOutlined style={{ fontSize: '24px', color: 'red' }} />
                      </Button>
                      <Button
                        type="text"
                        style={{ padding: '3px 6px' }}
                        disabled={!insertable}
                        onClick={() => {
                          addLyricWord()
                        }}
                      >
                        <CheckCircleOutlined
                          style={{
                            fontSize: '24px',
                            color: `${insertable ? 'green' : '#d9d9d9'}`,
                          }}
                        />
                      </Button>
                    </Space>
                  ))}
                </>
              )}
            </Form.List>
            <Form.Item label={'所在行'}>
              <Space>
                <Form.Item style={{ marginBottom: 0 }}>
                  <Input value={lyricTimeStamp[selectIndex.lyricIndex].text}></Input>
                </Form.Item>
                {/* <Button type="primary">插入字</Button> */}

                <Popconfirm
                  placement="rightTop"
                  title="确认删除这行吗?"
                  description="确认删除"
                  okText="删除"
                  cancelText="取消"
                  onConfirm={() => {
                    deleteLyricRow(selectIndex)
                  }}
                >
                  <Button danger icon={<DeleteOutlined />}>
                    删除该行
                  </Button>
                </Popconfirm>
              </Space>
            </Form.Item>
          </Form>
        ) : null}

        <Flex gap="middle">
          <InsertLyric />
        </Flex>
      </Flex>
    </>
  )
}

export default LyricOperation
