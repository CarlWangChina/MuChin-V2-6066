import json
import threading
import uuid
from datetime import datetime
from flask import request
from sqlalchemy import label, or_, func
from mapy import db
from . import examinerApi
from .model.examiner_work_model import ExaminerWork, ExaminerQaWork
from .model.result_model import Result
from ..investigate.model.answer_ui_model import AnswerUi
from ..investigate.model.options_ui_model import OptionUi
from ..investigate.model.question_model import Question
from ..music.model.language_model import Language
from ..music.model.music_model import Music
from ..music.model.work_model import WorkMusic, WorkQa, UserMusicAnswerResult
from ..openai.openai_api import gpt_lrc_thread
from ..result.network import success, fail
from ..user.models.user_model import User
from ..user.views import check_user_can_used
from ..util.other import from_session_map_cn

@examinerApi.route("/check_page/<work_id>/<token>")
def query_check_page_info(work_id: str, token: str):
    try:
        user = check_user_can_used(token)
    except Exception as err:
        return fail(str(err))
    if user['musician'] not in [3, 4]:
        return fail("权限不足")
    works = (db.session.query(ExaminerWork,
                              WorkMusic.song_lrc, WorkMusic.section, WorkMusic.rhyme, WorkMusic.extra_lan,
                              WorkMusic.finish_time,
                              User.id, User.name,
                              label("m_name", Music.name), Music.full_music)
             .join(WorkMusic, or_(ExaminerWork.work_id_1 == WorkMusic.id, ExaminerWork.work_id_2 == WorkMusic.id))
             .join(User, User.id == WorkMusic.uid)
             .join(Music, Music.id == ExaminerWork.mid)
             .filter(ExaminerWork.id == work_id,
                     ExaminerWork.is_finish == 0)
             .all())
    if len(works) == 0:
        return fail("获取质检歌曲失败")
    result = []
    for v in works:
        obj = {
            'id': v.ExaminerWork.id,
            'user': v.User.name,
            'time': v.WorkMusic.finish_time.strftime("%Y-%m-%d %H:%M:%S")
        }
        info = []
        if v.WorkMusic.extra_lan is None or v.WorkMusic.extra_lan == "":
            lrc_list = v.WorkMusic.song_lrc.split('\n')
            for i, line in enumerate(lrc_list):
                sub_sec = v.WorkMusic.section[i * 2: i * 2 + 2]
                sub_sec_desc = from_session_map_cn(sub_sec[0]) + str(ord(sub_sec[1]) - 96)
                ry_obj = {
                    'words': line,
                    'type': sub_sec_desc,
                }
                if v.WorkMusic.rhyme[i] != 'Z':
                    ry_obj['rhyme'] = v.WorkMusic.rhyme[i]
                info.append(ry_obj)
        else:
            lans = v.WorkMusic.extra_lan.split(',')
            q_lans = Language.query.filter(Language.id.in_(lans)).all()
            new_lans = []
            for i in q_lans:
                new_lans.append(i.lan)
            obj['lans'] = new_lans
        obj['info'] = info
        result.append(obj)
    return success({
        'name': works[0].m_name,
        'music': works[0].full_music,
        'compare_info': result
    })

@examinerApi.route("/check/<song_id>/<token>", methods=['POST'])
def create_user_answer_info(song_id: str, token: str):
    try:
        user = check_user_can_used(token)
    except Exception as err:
        return fail(str(err))
    if user['musician'] not in [3, 4]:
        return fail("权限不足")
    data = json.loads(request.data)
    error_list = data['error']
    right_list = data['right']
    if len(right_list) == 2:
        work = WorkMusic.query.filter_by(uid=user['id'], mid=song_id, is_finish=0).first()
        if work is None:
            new = WorkMusic(
                uid=user['id'],
                mid=song_id,
                song_lrc='',
                section='',
                rhyme='',
                get_time=datetime.now()
            )
            db.session.add(new)
        else:
            work.song_lrc = ''
            work.section = ''
            work.rhyme = ''
            work.get_time = datetime.now()
        db.session.commit()
        return success({
            'return': 0
        })
    elif len(right_list) == 1:
        db.session.add(UserMusicAnswerResult(
            uid=error_list[0],
            mid=song_id,
            state=3,
            create_time=datetime.now()
        ))
        db.session.add(UserMusicAnswerResult(
            uid=right_list[0],
            mid=song_id,
            state=1,
            create_time=datetime.now()
        ))
        try:
            copy_form_work = WorkMusic.query.filter_by(
                mid=song_id,
                uid=right_list[0]
            ).first()
        except Exception:
            return fail("请勿重复提交")
        if copy_form_work.extra_lan != '':
            lrc_extra_lan_state = 2
        else:
            lrc_extra_lan_state = 1
        new = Result(
            id=str(uuid.uuid4()).replace('-', ''),
            mid=song_id,
            type=1,
            lrc_song=copy_form_work.song_lrc,
            lrc_section=copy_form_work.section,
            lrc_rhyme=copy_form_work.rhyme,
            lrc_extra_lan=copy_form_work.extra_lan,
            user_from=user['id'] + "," + right_list[0],
            time=datetime.now(),
            ex_uid=user['id'],
            lrc_extra_lan_state=lrc_extra_lan_state
        )
        db.session.add(new)
        ex_work = ExaminerWork.query.filter_by(mid=song_id, ex_uid=user['id'], editable=2, is_finish=0).first()
        if ex_work is None:
            return fail("数据提交失败，请重新尝试 ,code:01")
        ex_work.is_finish = 1
        ex_work.finish_time = datetime.now()
        if new.lrc_extra_lan is None or new.lrc_extra_lan == '':
            create_qa_task_to_db(song_id, copy_form_work.song_lrc)
        db.session.commit()
        return success({
            'return': 1
        })
    else:
        copy_form_work = WorkMusic.query.filter_by(
            mid=song_id,
            uid=error_list[0]
        ).first()
        work = WorkMusic.query.filter_by(
            mid=song_id,
            uid=user['id']
        ).first()
        if work is None:
            new = WorkMusic(
                uid=user['id'],
                mid=song_id,
                song_lrc=copy_form_work.song_lrc,
                section=copy_form_work.section,
                rhyme=copy_form_work.rhyme,
                extra_lan='',
                get_time=datetime.now()
            )
            db.session.add(new)
        else:
            work.song_lrc = copy_form_work.song_lrc
            work.section = copy_form_work.section
            work.rhyme = copy_form_work.rhyme
            work.extra_lan = ''
            work.get_time = datetime.now()
        db.session.commit()
        if copy_form_work.extra_lan is None or copy_form_work.extra_lan == '':
            lan = 0
        else:
            lan = 1
        return success({
            'return': 0,
            'lan': lan
        })

@examinerApi.route("/detail/save/<submit_type>/<song_id>/<token>", methods=['POST'])
def check_music_info_and_save(submit_type: str, song_id: str, token: str):
    try:
        user = check_user_can_used(token)
    except Exception as err:
        return fail(str(err))
    if user['musician'] not in [3, 4]:
        return fail("权限不足")
    work = WorkMusic.query.filter_by(uid=user['id'], mid=song_id, is_finish=0).first()
    if work is None:
        return fail("获取歌曲信息失败，请重新尝试。code:11")
    data = json.loads(request.data)
    rhyme = ""
    for i in data:
        if i == '':
            rhyme += 'Z'
        else:
            rhyme += i
    work.rhyme = rhyme
    work.is_finish = 1
    work.finish_time = datetime.now()
    ex_work = ExaminerWork.query.filter_by(ex_uid=user['id'], mid=song_id, editable=2, is_finish=0).first()
    if ex_work is None:
        return fail("质检信息获取失败")
    ex_work.is_finish = 1
    ex_work.finish_time = datetime.now()
    query_result = Result.query.filter(Result.mid == song_id, Result.type == 1).first()
    users = db.session.query(WorkMusic.uid).filter(WorkMusic.mid == song_id, WorkMusic.uid != user['id']).all()
    if query_result is None:
        if submit_type == '2':
            lrc_from = users[0].uid + ',' + users[1].uid + ',' + user['id']
        else:
            lrc_from = user['id']
        result = Result(
            id=str(uuid.uuid4()).replace('-', ''),
            mid=song_id,
            type=1,
            lrc_song=work.song_lrc,
            lrc_section=work.section,
            lrc_rhyme=work.rhyme,
            user_from=lrc_from,
            time=datetime.now(),
            lrc_extra_lan_state=1,
            ex_uid=user['id']
        )
        db.session.add(result)
        db.session.commit()
        user_task_list = []
        if submit_type == '0':
            for u in users:
                user_task_list.append(
                    UserMusicAnswerResult(
                        uid=u.uid,
                        mid=song_id,
                        state=3,
                        create_time=datetime.now()
                    )
                )
        elif submit_type == '2':
            for u in users:
                user_task_list.append(
                    UserMusicAnswerResult(
                        uid=u.uid,
                        mid=song_id,
                        state=2,
                        create_time=datetime.now()
                    )
                )
        db.session.add_all(user_task_list)
        db.session.commit()
        create_qa_task_to_db(song_id, work.song_lrc)
        db.session.commit()
    return success({})

@examinerApi.route("/detail/save2/<submit_type>/<song_id>/<token>", methods=['POST'])
def check_music_info_and_save_new(submit_type: str, song_id: str, token: str):
    try:
        user = check_user_can_used(token)
    except Exception as err:
        return fail(str(err))
    if user['musician'] not in [3, 4]:
        return fail("权限不足")
    work = WorkMusic.query.filter_by(uid=user['id'], mid=song_id, is_finish=0).first()
    if work is None:
        return fail("获取歌曲信息失败，请重新尝试。code:11")
    data = json.loads(request.data)
    try:
        from Code_for_MuChin_AP.backend_api.mapy.music.views import handle_music_work_info
        handle_music_work_info(data, work)
    except Exception as err:
        return fail(str(err))
    ex_work = ExaminerWork.query.filter_by(ex_uid=user['id'], mid=song_id, editable=2, is_finish=0).first()
    if ex_work is None:
        return fail("质检信息获取失败")
    ex_work.is_finish = 1
    ex_work.finish_time = datetime.now()
    query_result = Result.query.filter(Result.mid == song_id, Result.type == 1).first()
    users = db.session.query(WorkMusic.uid).filter(WorkMusic.mid == song_id, WorkMusic.uid != user['id']).all()
    if query_result is None:
        if submit_type == '2':
            lrc_from = users[0].uid + ',' + users[1].uid + ',' + user['id']
        elif submit_type == '0':
            lrc_from = user['id']
        else:
            return fail("参数错误")
        result = Result(
            id=str(uuid.uuid4()).replace('-', ''),
            mid=song_id,
            type=1,
            lrc_song=work.song_lrc,
            lrc_section=work.section,
            lrc_rhyme=work.rhyme,
            user_from=lrc_from,
            time=datetime.now(),
            lrc_extra_lan_state=1,
            ex_uid=user['id']
        )
        db.session.add(result)
        db.session.commit()
        user_task_list = []
        if submit_type == '0':
            for u in users:
                user_task_list.append(
                    UserMusicAnswerResult(
                        uid=u.uid,
                        mid=song_id,
                        state=3,
                        create_time=datetime.now()
                    )
                )
        elif submit_type == '2':
            for u in users:
                user_task_list.append(
                    UserMusicAnswerResult(
                        uid=u.uid,
                        mid=song_id,
                        state=2,
                        create_time=datetime.now()
                    )
                )
        db.session.add_all(user_task_list)
        db.session.commit()
        create_qa_task_to_db(song_id, work.song_lrc)
        db.session.commit()
    return success({})

def create_qa_task_to_db(song_id: str, lrc: str):
    new_work1 = WorkQa(
        mid=song_id,
        special=0,
        create_time=datetime.now()
    )
    new_work2 = WorkQa(
        mid=song_id,
        special=1,
        create_time=datetime.now()
    )
    db.session.add(new_work1)
    db.session.add(new_work2)
    db.session.commit()
    thread = threading.Thread(target=gpt_lrc_thread, args=(song_id, lrc,))
    thread.start()

@examinerApi.route("/check/investigate/detail/<work_id>/<token>", methods=['GET'])
def query_check_investigate_page_info(work_id: str, token: str):
    try:
        user = check_user_can_used(token)
    except Exception as err:
        return fail(str(err))
    if user['musician'] not in [3, 30]:
        return fail("权限不足")
    info = (db.session.query(ExaminerQaWork.mid, WorkQa.uid, WorkQa.answer, User.musician)
            .join(WorkQa, WorkQa.id == ExaminerQaWork.work_id)
            .join(User, User.id == WorkQa.uid)
            .filter(ExaminerQaWork.id == work_id)
            .first())
    if info is None:
        return fail("该任务不存在。code:31")
    music = Music.query.filter_by(id=info.mid).first()
    result_music = Result.query.filter_by(type=1, mid=info.mid).first()
    qa_query_list = (Question.query
                     .filter_by(musician=info.musician)
                     .where(Question.sort > 0)
                     .order_by(Question.sort)
                     .all())
    format_answer = json.loads(info.answer)
    result = []
    for i, item in enumerate(qa_query_list):
        ui_list = []
        an_query_list = AnswerUi.query.filter_by(qid=item.id).order_by(AnswerUi.sort).all()
        for j, an in enumerate(an_query_list):
            obj = {
                'type': an.type,
                'title': an.title,
                'source': an.source
            }
            if an.exam_tips is None:
                obj['tips'] = ''
            else:
                obj['tips'] = an.exam_tips
            if an.type == '10' or an.type == '11':
                tags = format_answer[i]['tags'][j]['value']
                obj['tags'] = tags
                if 'desc' in format_answer[i]['tags'][j]:
                    obj['desc'] = format_answer[i]['tags'][j]['desc']
            elif an.type == '20' or an.type == '21':
                option_query_list = OptionUi.query.filter_by(aid=an.id)
                option_list = []
                for op in option_query_list:
                    option_list.append(op.format())
                obj['option_check'] = format_answer[i]['options'][j]['opid']
                obj['options'] = option_list
                if 'desc' in format_answer[i]['options'][j]:
                    obj['desc'] = format_answer[i]['options'][j]['desc']
            elif an.type == '0':
                obj['desc'] = format_answer[i]['desc']
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
                'title': "描述",
                'desc': format_answer[len(format_answer) - 1]['desc']
            }
        ]
    })
    return success(data={
        'music': music.abs_music,
        'name': music.name,
        'lrc': result_music.lrc_song,
        'qa': result
    })

@examinerApi.route("/check/investigate/grade/<work_id>/<token>", methods=['POST'])
def investigate_grade_save(work_id: str, token: str):
    try:
        user = check_user_can_used(token)
    except Exception as err:
        return fail(str(err))
    if user['musician'] not in [3, 30]:
        return fail("权限不足")
    data = json.loads(request.data)
    examiner_work: ExaminerQaWork = ExaminerQaWork.query.filter(ExaminerQaWork.id == work_id).first()
    if examiner_work is None:
        return fail("数据提交失败，请重新尝试 ,code:1 ")
    grade = ""
    grade_count = 0
    for i in data:
        grade += str(i) + ','
        grade_count += i
    examiner_work.qa_grade = grade[:-1]
    examiner_work.qa_grade_count = grade_count
    examiner_work.is_finish = 1
    examiner_work.finish_time = datetime.now()
    db.session.commit()
    return success({})

@examinerApi.route("/new/task/<token>", methods=['GET'])
def examiner_get_new_task(token: str):
    try:
        user = check_user_can_used(token)
    except Exception as err:
        return fail(str(err))
    if user['musician'] == 4:
        return fail("无需领取")
    elif user['musician'] not in [3, 30]:
        return fail("权限不足")
    limit = 20
    if user['power'] == 1:
        count_query = (db.session.query(label("count", func.count(ExaminerWork.ex_uid)))
                       .filter(ExaminerWork.ex_uid == user['id'],
                               ExaminerWork.is_finish == 0)
                       .first())
        need_add_count = limit - count_query.count
        if need_add_count <= 0:
            return success({})
        sql_str = f"select * from examiner_work where is_finish = 0 and editable = 2 ORDER BY RAND() LIMIT {need_add_count}"
        db.session.execute(db.text(sql_str))
        db.session.commit()
        examiner_task_list = (db.session.query(Music.id, Music.name, ExaminerWork)
                              .join(ExaminerWork, ExaminerWork.mid == Music.id)
                              .filter_by(is_finish=0, ex_uid=user['id'])
                              .all())
        result = []
        for value in examiner_task_list:
            result.append({
                'work_id': value.ExaminerWork.id,
                'music_id': value.id,
                'music_name': value.name,
                'editable': 2
            })
        return success({
            'song_list': result
        })
    elif user['power'] == 2:
        count_query = (db.session.query(label("count", func.count(ExaminerQaWork.ex_uid)))
                       .filter(ExaminerQaWork.ex_uid == user['id'],
                               ExaminerQaWork.is_finish == 0)
                       .first())
        need_add_count = limit - count_query.count
        if need_add_count <= 0:
            return success({})
        if user['musician'] == 30:
            sql_str = f"select * from examiner_qa_work where is_finish = 0 and special = 1 ORDER BY RAND() LIMIT {need_add_count}"
        else:
            sql_str = f"select * from examiner_qa_work where is_finish = 0 and special = 0 ORDER BY RAND() LIMIT {need_add_count}"
        db.session.execute(db.text(sql_str))
        db.session.commit()
        examiner_task_list = (db.session.query(Music.id, Music.name, ExaminerQaWork)
                              .join(ExaminerQaWork, ExaminerQaWork.mid == Music.id)
                              .filter_by(is_finish=0, ex_uid=user['id'])
                              .all())
        result = []
        for value in examiner_task_list:
            if value.ExaminerQaWork.special == 0:
                music_name = f'{value.name} (非专业)'
            else:
                music_name = f'{value.name} (专业)'
            result.append({
                'work_id': value.ExaminerQaWork.id,
                'music_id': value.id,
                'music_name': music_name,
                'editable': 1
            })
        return success({
            'song_list': result
        })
    else:
        return fail("未知错误")

@examinerApi.route("/check/pending/<work_id>/<token>", methods=['POST'])
def examiner_pending_music(work_id: str, token: str):
    try:
        user = check_user_can_used(token)
    except Exception as err:
        return fail(str(err))
    if user['musician'] == 4:
        return fail("该账号不支持挂起")
    elif user['musician'] != 3:
        return fail("权限不足")
    reason_obj = json.loads(request.data)
    examiner_task = ExaminerWork.query.filter_by(id=work_id, ex_uid=user['id'], is_finish=0).first()
    if examiner_task is not None:
        examiner_task.is_finish = 2
        if 'reason' in reason_obj:
            examiner_task.pending_reason = reason_obj['reason']
        examiner_task.finish_time = datetime.now()
        special_examiner = find_special_examiner()
        if special_examiner is not None and examiner_task.editable == 2:
            special_examiner_task = ExaminerWork(
                id=str(uuid.uuid4()).replace('-', ''),
                ex_uid=special_examiner.id,
                mid=examiner_task.mid,
                work_id_1=examiner_task.work_id_1,
                work_id_2=examiner_task.work_id_2,
                editable=2,
                create_time=datetime.now(),
                get_time=datetime.now()
            )
            db.session.add(special_examiner_task)
        db.session.commit()
        return success({})
    return fail("标注歌曲不存在")

def find_special_examiner():
    return User.query.filter_by(musician=4, delete=0).first()