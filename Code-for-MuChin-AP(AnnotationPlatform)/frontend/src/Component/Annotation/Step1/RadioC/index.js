import { Radio } from 'antd'

const RadioC = ({ rValue, options, onChange, pe }) => {
  return (
    <div style={{ width: '100%' }}>
      <Radio.Group
        className="que_ui"
        disabled={pe === 'none'}
        onChange={onChange}
        value={rValue}
      >
        {options.map((op) => (
          <Radio key={op.id} value={op.id}>
            {op.name}
          </Radio>
        ))}
      </Radio.Group>
    </div>
  )
}

export default RadioC