import { MinusCircleOutlined, PlusOutlined } from '@ant-design/icons'
import { Alert, Button, Drawer, Form, FormInstance, Input, message, Space } from 'antd'
import React, { useEffect, useState } from 'react'

import { useLyricStore } from '../../../../../stores/LrcTimestamp'

interface SubmitButtonProps {
  form: FormInstance
  onSubmit: () => { successed: boolean; info: string }
}

export const SubmitButton: React.FC<React.PropsWithChildren<SubmitButtonProps>> = ({
  form,
  onSubmit,
  children,
}) => {
  const [submittable, setSubmittable] = React.useState<boolean>(false)

  // Watch all values
  const values = Form.useWatch([], form)

  const [messageApi, contextHolder] = message.useMessage()

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

  React.useEffect(() => {
    form
      .validateFields({ validateOnly: true })
      .then(() => setSubmittable(true))
      .catch(() => setSubmittable(false))
  }, [form, values])

  return (
    <>
      {contextHolder}
      <Space>
        <Button
          type="primary"
          htmlType="submit"
          disabled={!submittable}
          onClick={() => {
            const res = onSubmit()
            if (res.successed) {
              success(res.info)
            } else {
              error(res.info)
            }
          }}
        >
          {children}
        </Button>
      </Space>
    </>
  )
}

const InsertLyric: React.FC = () => {
  const [form] = Form.useForm()
  const [open, setOpen] = useState(false)
  const [lyricText, setLyricText] = useState('')
  const insertLyricRow = useLyricStore((state) => {
    return state.insertLyricRow
  })

  const showDrawer = () => {
    setOpen(true)
  }

  const onClose = () => {
    setOpen(false)
  }
  useEffect(() => {}, [])

  const onFeildChange = () => {
    const lyrics = form.getFieldValue('lyricRow')
    const text: [] = lyrics.map((v) => {
      if (v !== undefined) {
        return v['word']
      }
    })
    console.log(lyrics)
    setLyricText(text.join(''))
  }

  function submitLyricAdd() {
    const words = form.getFieldValue('lyricRow')
    let checked = true
    let rowText = ''

    const rowWords: (string | number)[][] = words.map((word, index, array) => {
      const start = stringTimeToSeconds(word['start'])
      const end = stringTimeToSeconds(word['end'])
      const wordText: string = word['word']
      if (start >= end) {
        checked = false
      }
      if (index > 0 && start < stringTimeToSeconds(array[index - 1]['end'])) {
        checked = false
      }
      rowText += word['word']
      return [start * 1000, end * 1000, wordText]
    })
    if (!checked) {
      return { successed: false, info: '歌词行内时间戳大小校验不通过' }
    }
    // 插入原歌词校验
    const rowStartStr: string = words[0]['start']
    const rowEndStr: string = words[words.length - 1]['end']
    const rowStart = stringTimeToSeconds(rowStartStr) * 1000
    const rowEnd = stringTimeToSeconds(rowEndStr) * 1000
    const staus = insertLyricRow(rowText, rowStart, rowEnd, rowWords)
    if (!staus) {
      return { successed: false, info: '歌词行插入位置时间戳校验不通过' }
    } else {
      setLyricText('')
      form.resetFields()
      return { successed: true, info: '歌词行添加成功' }
    }
  }

  function stringTimeToSeconds(time1: string) {
    const time1Value = time1.split(':').reduce((minutes, seconds) => {
      return (Number(minutes) * 60 + Number(seconds)).toString()
    })
    return Number(time1Value)
  }

  return (
    <>
      <Button type="primary" onClick={showDrawer} icon={<PlusOutlined />}>
        添加歌词行
      </Button>
      <Drawer
        title="插入歌词行"
        width="600"
        onClose={onClose}
        open={open}
        styles={{
          body: {
            paddingBottom: 80,
          },
        }}
        extra={
          <Space>
            <Button
              htmlType="reset"
              onClick={() => {
                form.resetFields()
              }}
            >
              Reset
            </Button>
            <SubmitButton form={form} onSubmit={submitLyricAdd}>
              Submit
            </SubmitButton>
          </Space>
        }
      >
        <Form
          form={form}
          name="lyric-insert"
          autoComplete="off"
          variant="filled"
          style={{ maxWidth: 500 }}
        >
          <Form.Item label="歌词行">
            <Input
              placeholder="歌词行"
              disabled
              value={lyricText}
              maxLength={1}
              variant="borderless"
            />
          </Form.Item>
          <Form.List name="lyricRow">
            {(fields, { add, remove }) => (
              <>
                {fields.map(({ key, name, ...restField }) => (
                  <Space key={key} style={{ display: 'flex', marginBottom: 8 }} align="baseline">
                    <Form.Item
                      {...restField}
                      name={[name, 'word']}
                      rules={[{ required: true, message: '输入一个字' }]}
                    >
                      <Input
                        placeholder="歌词字"
                        maxLength={1}
                        onPressEnter={() => {
                          onFeildChange()
                        }}
                        onBlur={() => {
                          onFeildChange()
                        }}
                      />
                    </Form.Item>
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
                    <MinusCircleOutlined onClick={() => remove(name)} />
                  </Space>
                ))}
                <Form.Item>
                  <Button type="dashed" onClick={() => add()} block icon={<PlusOutlined />}>
                    新增字
                  </Button>
                </Form.Item>
              </>
            )}
          </Form.List>
        </Form>
      </Drawer>
    </>
  )
}

export default InsertLyric
