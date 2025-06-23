import json
from datetime import datetime
from flask import request
from sqlalchemy import label, func
from mapy import db
from mapy.examiner.model.examiner_work_model import ExaminerQaWork
from . import qaApi
from .model.answer_ui_model import AnswerUi
from .model.options_ui_model import OptionUi
from .model.question_model import Question
from ..examiner.model.result_model import Result
from ..music.model.music_model import Music
from ..music.model.work_model import WorkQa
from ..result.network import success, fail
from Code_for_MuChin_AP.backend_api.mapy.user.views import check_user_can_used

@qaApi.route("detail/<song_id>/<token>", methods=['GET', 'POST'])
def qa_detail(song_id: str, token: str):
    try:
        user = check_user_can_used(token)
    except Exception as err:
        return fail(str(err))
    if request.method == 'GET':
        query = (db.session.query(Music.id, Music.name, Music.abs_music, Result.lrc_song)
                 .join(Result, Result.mid == Music.id)
                 .filter(Music.id == song_id)
                 .first())
        qa_query_list = (Question.query
                         .filter_by(musician=user['musician'])
                         .where(Question.sort > 0)
                         .order_by(Question.sort)
                         .all())
        gpt_content = ""
        if user['power'] == 2:
            qa_info = WorkQa.query.filter_by(uid=user['id'], mid=song_id, is_finish=0).first()
            if qa_info is None:
                return fail("获取信息失败，请重新尝试。code:22")
            gpt_content = qa_info.gpt_sum_content
        if query is None or qa_query_list is None:
            return fail("获取信息失败，请重新尝试。code:21")
        result = []
        for item in qa_query_list:
            ui_list = []
            an_query_list = AnswerUi.query.filter_by(qid=item.id).order_by(AnswerUi.sort).all()
            for an in an_query_list:
                obj = {
                    'sub_id': an.id,
                    'type': an.type,
                    'title': an.title
                }
                if an.tips is None:
                    obj['tips'] = ''
                else:
                    obj['tips'] = an.tips
                if an.type == '10' or an.type == '11':
                    obj['category_id'] = an.cid
                    obj['min_len'] = an.ui_min_text_len
                    obj['min_tag'] = an.ui_min_tag_num
                    if an.content_need_gpt == 1:
                        obj['desc'] = gpt_content
                elif an.type == '20' or an.type == '21':
                    option_query_list = OptionUi.query.filter_by(aid=an.id).order_by().all()
                    option_list = []
                    for op in option_query_list:
                        option_list.append(op.format(add_must=(an.type == '21')))
                    obj['options'] = option_list
                else:
                    obj['min_len'] = an.ui_min_text_len
                ui_list.append(obj)
            result.append({
                'id': item.id,
                'index': item.sort,
                'question': item.question,
                'ui': ui_list
            })
        result.append({
            'id': '0',
            'index': len(result) + 1,
            'question': '其他(选填)',
            'ui': [
                {
                    'type': '0',
                    'must': '0',
                    'title': "描述",
                    'min_len': -1
                }
            ]
        })
        return success(data={
            'music': query.abs_music,
            'name': query.name,
            'lrc': query.lrc_song,
            'qa': result
        })
    if request.method == 'POST':
        work = WorkQa.query.filter_by(uid=user['id'], mid=song_id, is_finish=0).first()
        if work is None:
            return fail("信息获取失败，请重新尝试。code:22")
        data = json.loads(request.data)
        if len(data) == 0:
            return fail("请输入正确的内容")
        work.answer = json.dumps(data)
        work.is_finish = 1
        work.finish_time = datetime.now()
        db.session.commit()
        check_need_examiner_and_update(user, work)
        return success({})

def check_need_examiner_and_update(user, work: WorkQa):
    LIMIT = 5
    count_query = (db.session.query(label("count", func.count(WorkQa.uid)))
                   .filter(WorkQa.uid == user['id'], WorkQa.is_finish == 1).first())
    if count_query.count % LIMIT == 0:
        db.session.add(ExaminerQaWork(
            work_id=work.id,
            mid=work.mid,
            create_time=datetime.now(),
            special=work.special
        ))
        db.session.commit()

@qaApi.route("category/<cid>/<token>", methods=['GET'])
def query_category_list(cid: str, token: str):
    try:
        user = check_user_can_used(token)
    except Exception as err:
        return fail(str(err))
    if request.method == 'GET':
        table = 'wyy_class_word_frequency'
        if user['musician'] == 1:
            table = 'musiccaps_aspects'
        query_list = db.session.execute(
            db.text("select * from " + table + " where category_id & " + cid + " > 0")
        ).all()
        result = []
        for i in query_list:
            result.append(i.name_cn)
            if cid == '512':
                result.append('无明显用途')
        return success(result)

@qaApi.route("/pending/<song_id>/<token>", methods=['POST'])
def investigate_pending_music(song_id: str, token: str):
    try:
        user = check_user_can_used(token)
    except Exception as err:
        return fail(str(err))
    reason_obj = json.loads(request.data)
    qa_task = WorkQa.query.filter_by(mid=song_id, uid=user['id'], is_finish=0).first()
    if qa_task is not None:
        qa_task.is_finish = 2
        if 'reason' in reason_obj:
            qa_task.pending_reason = reason_obj['reason']
        qa_task.finish_time = datetime.now()
        db.session.commit()
        return success({})
    return fail("标注歌曲不存在")