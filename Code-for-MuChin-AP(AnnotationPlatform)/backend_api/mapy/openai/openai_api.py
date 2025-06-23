import os
from openai import OpenAI
import sys
import db
from Code_for_MuChin_AP_AnnotationPlatform_.backend_api.mapy.music.model.work_model import Work
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def api_req(lrc) -> str:
    if os.getenv('OS_ENVIRONMENT') == "production":
        sysct = usrct = f'[歌词]\n\n：{lrc}\n\n'
        response = client.chat.completions.create(
            model='gpt-3.5-turbo-1106',
            messages=[
                {'role': 'system', 'content': sysct},
                {'role': 'user', 'content': usrct}
            ]
        )
        return response.choices[0].message.content
    else:
        return "这首歌的主题似乎是关于渴望和需要某人的存在，因为他们被描述为“唯一”并且在没有他们的情况下将会“死去”。这些歌词充满着紧迫感和渴望，似乎反映了一种深刻的依赖关系或者对某人的强烈渴望。风格上，这首歌可能会以简单连贯的旋律和节奏，带有强烈的情感和对某人渴望的表达为主。"

def gpt_lrc_thread(mid: str, lrc: str):
    gpt_info = api_req(lrc)
    with app.application.app_context():
        work_qa_list = WorkQa.query.filter_by(mid=mid).all()
        for work_qa in work_qa_list:
            work_qa.gpt_sum_content = gpt_info
            db.session.commit()