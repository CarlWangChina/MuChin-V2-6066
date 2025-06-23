import { Button, Card, Popover } from 'antd'

const LrcShowC = ({ data, loading, onSubmitRight, onSubmitError }) => {
  return (
    <div>
      <Card
        title={data.user}
        extra={
          <ul style={{ margin: 0, padding: 0 }}>
            <li className="check_card_extra_li">
              <Popover
                content={
                  <div className="check_btn_tip">被选的那份明确记为正确，另一份明确记为错误</div>
                }
              >
                <Button type="primary" loading={loading} onClick={() => onSubmitRight()}>
                  该结果为正确版本
                </Button>
              </Popover>
            </li>
            <li className="check_card_extra_li">
              <Popover
                content={
                  <div>
                    <div className="check_btn_tip">在被选的那份基础之上进行修正，两份都记为错</div>
                    <div className="check_btn_tip">(两个人都错了，我选其中一个人继续改)</div>
                  </div>
                }
                placement="bottom"
              >
                <Button type="primary" loading={loading} onClick={() => onSubmitError()}>
                  对该结果进行修改
                </Button>
              </Popover>
            </li>
          </ul>
        }
        className="card_qa"
      >
        {data.info.length > 0 ? (
          data.info.map((item, index) => {
            return (
              <div key={index} style={{ display: 'flex', userSelect: 'none' }}>
                <div style={{ fontSize: 'large' }}>
                  <strong>{item.type}&nbsp;</strong>
                </div>
                <div style={{ fontSize: 'large' }}>{String(item.words).slice(0, -1)}</div>
                <div
                  style={{
                    color: item.bg,
                    fontSize: 'large',
                    fontWeight: item.rhyme === undefined || item.rhyme === '' ? 'normal' : 'bold',
                  }}
                >
                  {String(item.words).slice(-1)}
                </div>
              </div>
            )
          })
        ) : (
          <div style={{ fontSize: 'large' }}>
            <strong>该歌曲被跳过</strong>
            <div>原因如下：</div>
            <ul>
              {data.lans.map((item, index) => (
                <li style={{ margin: 0 }} key={index}>
                  {item}
                </li>
              ))}
            </ul>
          </div>
        )}
      </Card>
    </div>
  )
}

export default LrcShowC