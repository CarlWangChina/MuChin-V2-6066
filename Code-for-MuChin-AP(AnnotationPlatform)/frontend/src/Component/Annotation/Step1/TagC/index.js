import { Select } from 'antd'

const TagC = ({ tValue, options, minTags, onFocus, onChange, pe }) => {
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
        onChange={(e) => onChange(e)}
        value={tValue}
        placeholder={pe === undefined ? `请选择或输入至少${minTags}个标签` : undefined}
      />
    </div>
  )
}

export default TagC