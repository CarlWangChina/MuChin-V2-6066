import { Input } from 'antd'

const { TextArea } = Input

const EditC = ({ eValue, must, minLen, onPaste, onChange, pe }) => {
  return (
    <div style={{ width: '100%' }}>
      <TextArea
        disabled={pe === 'none'}
        className="que_ui_area"
        autoSize={{ minRows: 3 }}
        onPaste={(e) => onPaste(e)}
        autoComplete="off"
        maxLength={minLen === -1 ? undefined : 300}
        showCount={minLen === -1 ? false : true}
        value={eValue}
        onChange={(e) => onChange(e)}
        placeholder={
          pe === undefined
            ? must === '0'
              ? '请输入您的描述（非必填）'
              : minLen === -1
                ? '请输入您的描述（必填）'
                : `请输入至少${minLen}字的描述`
            : undefined
        }
      />
    </div>
  )
}

export default EditC