import { QuestionCircleOutlined } from '@ant-design/icons'
import { FloatButton } from 'antd'
import Base64 from 'base-64'
import { useNavigate } from 'react-router-dom'

import { pushCheckPendingReasonAsync } from '../../api/check'
import { getToken } from '../../utils/auth'
import Lyrics from '../Annotation/Lyrics'
import withRouter from '../Base/WithRouter/withRouter'
import Header from '../Header'
import LyricsShow from './LyricsShow'
import MusicShow from './MusicShow'

const Quality = (props) => {
  const navigate = useNavigate()

  const { is } = props.params

  const step_obj = JSON.parse(Base64.decode(is))
  const { id, step, type, work_id } = step_obj

  let StepC

  switch (step) {
    case 1:
      StepC = MusicShow
      break
    case 2:
      StepC = LyricsShow
      break
    case 3:
    case 4:
    case 5:
      StepC = Lyrics
      break
    default:
      break
  }
  const onQuestionClick = async () => {
    let rs_ques = ''
    while (true) {
      const ques = prompt('请输入挂起原因', '')
      if (ques === null) {
        break
      } else if (ques.length > 0) {
        rs_ques = ques
        break
      } else {
        alert('请输入挂起原因')
        continue
      }
    }
    if (rs_ques !== '') {
      try {
        const res = await pushCheckPendingReasonAsync(work_id, getToken(), { reason: rs_ques })
        if (Number(res.data.code) === 200) {
          navigate('/home')
        } else {
          alert(res.data.msg)
          return
        }
      } catch (err) {
        if (String(err).includes('timeout')) {
          alert('请求超时！')
        } else if (String(err).includes('500')) {
          alert('请求失败！')
        } else {
          alert(err)
        }
        return
      }
    }
  }
  return (
    <div className="annotation">
      <Header />
      <StepC id={id} type={type} work_id={work_id} />
      <FloatButton
        tooltip={<div>挂起搁置并反馈原因</div>}
        icon={<QuestionCircleOutlined />}
        onClick={() => onQuestionClick()}
      />
    </div>
  )
}

export default withRouter(Quality)