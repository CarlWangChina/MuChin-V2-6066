import { Button } from 'antd'

const List = ({ musics, onClickItem }) => {
  return musics.length > 0 ? (
    <div style={{ width: '100%' }}>
      {musics.map((item) => (
        <div
          key={item.work_id}
          className="check_list_item"
          style={{ display: 'flex', alignItems: 'center' }}
        >
          <div style={{ flex: 1, fontSize: 'large', fontWeight: 'bold' }}>{item.music_name}</div>
          <Button type="primary" onClick={() => onClickItem(item.type, item.work_id)}>
            查看
          </Button>
        </div>
      ))}
    </div>
  ) : (
    <h2>查不到相关数据</h2>
  )
}

export default List