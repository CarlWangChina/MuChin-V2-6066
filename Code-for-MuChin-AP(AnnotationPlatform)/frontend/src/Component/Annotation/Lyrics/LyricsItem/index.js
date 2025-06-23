import {
  CloseOutlined,
  EnterOutlined,
  MinusCircleTwoTone,
  PlusCircleTwoTone,
  PlusOutlined,
  RollbackOutlined,
} from '@ant-design/icons'
import { Button, Cascader, Input, Tooltip } from 'antd'

const LyricsItemC = ({
  item,
  tagOptions,
  onLyricsInputChange,
  onLyricsPressEnter,
  onPinyinInputChange,
  onPinyinPressEnter,
  onTimeInputChange,
  onTimePressEnter,
  onAddLine,
  onRemoveLine,
  onClickSplit,
  selectChange,
}) => {
  return (
    <div className="lyrics_item_container">
      <div style={{ display: 'flex', alignItems: 'center' }}>
        <Input
          className="lyrics_item_time"
          value={item.time_change.length > 0 ? item.time_change : item.time}
          onChange={(value) => onTimeInputChange(value)}
          onPressEnter={() => onTimePressEnter()}
          onBlur={() => onTimePressEnter()}
        />
        <Input
          className="lyrics_item_words"
          value={item.words}
          onChange={(value) => onLyricsInputChange(value)}
          onPressEnter={() => onLyricsPressEnter()}
        />
        <Input
          className="lyrics_item_pinyin"
          value={item.pinyin}
          style={{ color: item.bg }}
          onChange={(value) => onPinyinInputChange(value)}
          onPressEnter={(value) => onPinyinPressEnter(value)}
        />
        <Tooltip title="在这一行下新增一行">
          <Button
            type="primary"
            className="lyrics_item_add_btn"
            style={{ fontSize: 'large' }}
            onClick={() => onAddLine()}
          >
            <EnterOutlined />
            <PlusOutlined />
          </Button>
        </Tooltip>

        <Tooltip title="删除本行">
          <Button
            type="primary"
            className="lyrics_item_remove_btn"
            style={{ fontSize: 'large' }}
            onClick={() => onRemoveLine()}
          >
            <RollbackOutlined />
            <CloseOutlined />
          </Button>
        </Tooltip>

        <Tooltip title={`${item.sp !== true ? '新增' : '去除'}段落分割线`}>
          <div
            className="lyrics_item_sec_action"
            onClick={() => onClickSplit()}
            style={{ cursor: 'pointer', fontSize: 'large', userSelect: 'none' }}
          >
            {item.sp !== true ? (
              <PlusCircleTwoTone style={{ fontSize: 24 }} />
            ) : (
              <MinusCircleTwoTone style={{ fontSize: 24 }} />
            )}
          </div>
        </Tooltip>
      </div>
      {item.sp === true ? (
        <div style={{ display: 'flex', alignItems: 'center', paddingTop: 10, paddingBottom: 20 }}>
          <hr style={{ flex: 1 }} />
          <div>以上为：</div>
          <Cascader
            value={item.type}
            onChange={(value) => selectChange(value)}
            options={tagOptions}
            displayRender={(label) => label}
          />
        </div>
      ) : null}
    </div>
  )
}

export default LyricsItemC