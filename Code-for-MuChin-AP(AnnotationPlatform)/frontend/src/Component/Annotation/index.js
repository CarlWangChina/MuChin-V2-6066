import './index.scss'

import { QuestionCircleOutlined } from '@ant-design/icons'
import { FloatButton } from 'antd'
import Base64 from 'base-64'
import { useNavigate } from 'react-router-dom'

import { pushMusicPendingReasonAsync, pushPendingReasonAsync } from '../../api/lyrics'
import { getToken } from '../../utils/auth'
import withRouter from '../Base/WithRouter/withRouter'
import Header from '../Header'
import Step1 from './Step1'
import TimestampPage from './Timestamp'

const Annotation = (props) => {
  const navigate = useNavigate()

  const { is } = props.params

  const step_obj = JSON.parse(Base64.decode(is))
  const { id, step } = step_obj

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
        const res =
          Number(step) === 1
            ? await pushMusicPendingReasonAsync(id, getToken(), { reason: rs_ques })
            : await pushPendingReasonAsync(id, getToken(), { reason: rs_ques })
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
      {Number(step) === 1 ? <Step1 id={id} /> : <TimestampPage mid={id} type={0} />}

      <FloatButton
        tooltip={<div>挂起搁置并反馈原因</div>}
        icon={<QuestionCircleOutlined />}
        onClick={() => onQuestionClick()}
      />
    </div>
  )
}

export default withRouter(Annotation)