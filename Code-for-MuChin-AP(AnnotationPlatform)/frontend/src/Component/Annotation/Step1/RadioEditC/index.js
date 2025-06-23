import { Input, Radio } from 'antd'

const { TextArea } = Input

const RadioEditC = ({ rValue, eValue, options, onChangeR, onChangeE, onPaste, pe }) => {
  return (
    <div style={{ width: '100%' }}>
      {pe === undefined ? (
        <div className="radio_edit_tip">{`如果选择“${options.find((item) => item.must === '1').name}”，请输入至少${options.find((item) => item.must === '1').min_len}字描述`}</div>
      ) : null}

      <Radio.Group
        className="que_ui"
        disabled={pe === 'none'}
        onChange={onChangeR}
        value={rValue}
      >
        {options.map((op) => {
          return (
            <Radio key={op.id} value={op.id}>
              {op.name}
            </Radio>
          )
        })}
      </Radio.Group>

      <TextArea
        className="que_ui_area"
        disabled={pe === 'none'}
        autoSize={'true'}
        onPaste={(e) => onPaste(e)}
        autoComplete="off"
        maxLength={300}
        showCount
        value={eValue}
        onChange={(e) => onChangeE(e)}
      />
    </div>
  )
}

export default RadioEditC