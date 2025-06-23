import { Input, Select } from 'antd'

const { TextArea } = Input

const TagEditC = ({
  tValue,
  eValue,
  minLen,
  minTags,
  options,
  onFocus,
  onChangeE,
  onChangeS,
  onPaste,
  pe,
}) => {
  return (
    <div style={{ width: '100%' }}>
      <Select
        className="que_ui"
        style={{ width: '100%' }}
        disabled={pe === 'none'}
        mode="tags"
        tokenSeparators={[',', '，']}
        options={options}
        onFocus={() => onFocus()}
        onChange={(e) => onChangeS(e)}
        value={tValue}
        placeholder={pe === undefined ? `请选择或输入至少${minTags}个标签` : undefined}
      />
      <TextArea
        className="que_ui_area"
        disabled={pe === 'none'}
        autoSize={{ minRows: 3 }}
        onPaste={(e) => onPaste(e)}
        autoComplete="off"
        maxLength={minLen === -1 ? undefined : 300}
        showCount={minLen === -1 ? false : true}
        value={eValue}
        onChange={(e) => onChangeE(e)}
        placeholder={
          pe === undefined
            ? minLen === -1
              ? '请输入您的描述（必填）'
              : `请输入至少${minLen}字的描述`
            : undefined
        }
      />
    </div>
  )
}

export default TagEditC