import json
import jwt
from sqlalchemy import label, and_
from config import Config
from . import managerApi
import managerApi
from .. import db
from ..examiner.model.result_model import Result
from ..investigate.model.answer_ui_model import AnswerUi
import AnswerUi
from ..investigate.model.options_ui_model import OptionUi
import OptionUi
from ..investigate.model.question_model import Question
from ..music.model.language_model import Language
from ..music.model.music_model import Music
from ..music.model.work_model import WorkMusic, WorkMusicTemp, WorkQa
import WorkMusic, WorkMusicTemp, WorkQa
from ..result.network import success, fail
from ..user.models.user_model import User
from Code_for_MuChin_AP_AnnotationPlatform_.backend_api.mapy.util.other import from_session_map_cn

@managerApi.route('search/<user>/<token>', methods=['GET'])
def search_user_music_info(user: str, token: str):
    admin = jwt.decode(token, Config.TOKEN_SECRET, algorithms=['HS256'])
    if user == "":
        return success({})
    user = User.query.filter_by(name=user).first()
    if user is None:
        return fail("该用户不存在")
    result = []
    if user.power == 1:
        if user.musician != 3:
            query_list = (db.session.query(
                Music.id, Music.name, WorkMusic
            ).join(Music, Music.id == WorkMusic.mid)
                          .filter(
                WorkMusic.uid == user.id,
                WorkMusic.is_finish != 0
            ).all())
            for v in query_list:
                result.append({
                    'work_id': str(v.WorkMusic.id),
                    'music_name': v.name,
                    'type': 1
                })
            query_temp_list = (db.session.query(
                Music.id, Music.name, WorkMusicTemp
            ).join(Music, Music.id == WorkMusicTemp.mid)
                               .filter(
                WorkMusicTemp.uid == user.id,
                WorkMusicTemp.is_finish != 0
            ).all())
            for v in query_temp_list:
                result.append({
                    'work_id': str(v.WorkMusicTemp.id),
                    'music_name': v.name,
                    'type': 1
                })
            return success({
                'song_list': result,
                'uid': user.id
            })
        else:
            query_list = (db.session.query(
                Music.id, Music.name, Result
            ).join(Music, Music.id == Result.mid)
            .filter(
                Result.user_from.ilike(f"%{user.id}%")
            )).all()
            result = []
            for v in query_list:
                result.append({
                    'work_id': str(v.Result.id),
                    'music_name': v.name,
                    'type': 1
                })
            return success({
                'song_list': result,
                'uid': user.id
            })
    elif user.power == 2:
        if user.musician != 3:
            query_list = (db.session.query(
                Music.id, Music.name, WorkQa
            ).join(Music, Music.id == WorkQa.mid)
                          .filter(
                WorkQa.uid == user.id,
                WorkQa.is_finish != 0
            ).all())
            for v in query_list:
                result.append({
                    'work_id': str(v.WorkQa.id),
                    'music_name': v.name,
                    'type': 2
                })
            return success({
                'song_list': result,
                'uid': user.id
            })
        else:
            query_list = (db.session.query(
                Music.id, Music.name, Result
            ).join(Music, Music.id == Result.mid)
                          .filter(
                Result.ex_uid == user.id,
                Result.type == 2
            ).all())
            for v in query_list:
                result.append({
                    'work_id': str(v.Result.id),
                    'music_name': v.name,
                    'type': 2
                })
            return success({
                'song_list': result,
                'uid': user.id
            })
    else:
        return fail("未知错误")

@managerApi.route('detail/<uid>/<work_id>/<token>', methods=['GET'])
def lrc_detail_watching(uid: str, work_id: str, token: str):
    admin = jwt.decode(token, Config.TOKEN_SECRET, algorithms=['HS256'])
    if uid == "":
        return fail("用户信息错误")
    user = User.query.filter_by(id=uid).first()
    if user is None:
        return fail("用户不存在")
    if user.power != 1:
        return fail("权限不足")
    temp = {}
    if user.musician == 3:
        query = (db.session.query(
            Music.name, Music.full_music, Result
        ).join(Result, Result.mid == Music.id)
                 .filter(Result.id == work_id).first())
        temp['song_url'] = query.full_music
        temp['song_name'] = query.name
        temp['lrc'] = query.Result.lrc_song
        temp['section'] = query.Result.lrc_section
        temp['rhyme'] = query.Result.lrc_rhyme
        temp['lan'] = query.Result.lrc_extra_lan
    else:
        query = (db.session.query(
            Music.name, Music.full_music,
            label("lrc", WorkMusic.song_lrc),
            label("section", WorkMusic.section),
            label("rhyme", WorkMusic.rhyme),
            label("lan", WorkMusic.extra_lan),
            label("pending", WorkMusic.pending_reason),
        ).join(WorkMusic, WorkMusic.mid == Music.id)
                 .filter(WorkMusic.id == work_id).first())
        if query is None:
            query = (db.session.query(
                Music.name, Music.full_music,
                label("lrc", WorkMusicTemp.song_lrc),
                label("section", WorkMusicTemp.section),
                label("rhyme", WorkMusicTemp.rhyme),
                label("lan", WorkMusicTemp.extra_lan),
                label("pending", WorkMusicTemp.pending_reason),
            ).join(WorkMusicTemp, WorkMusicTemp.mid == Music.id)
                     .filter(WorkMusicTemp.id == work_id).first())
        temp['song_url'] = query.full_music
        temp['song_name'] = query.name
        if query.pending != '':
            temp['pending'] = query.pending
            return success(temp)
        temp['lrc'] = query.lrc
        temp['section'] = query.section
        temp['rhyme'] = query.rhyme
        temp['lan'] = query.lan
        if temp['lan'] != '':
            lans = temp['lan'].split(',')
            q_lans = Language.query.filter(Language.id.in_(lans)).all()
            new_lans = []
            for i in q_lans:
                new_lans.append(i.lan)
            return success({
                'song_url': temp['song_url'],
                'song_name': temp['song_name'],
                'reason': new_lans
            })
    result = []
    lrc_split = temp['lrc'].split('\n')
    for i, v in enumerate(lrc_split):
        cur_section = temp['section'][i * 2: i * 2 + 2]
        obj = {
            'line': i,
            'type': from_session_map_cn(cur_section[0]) + str(ord(cur_section[1]) - 96),
            'words': v
        }
        if temp['rhyme'][i] != 'Z':
            obj['rhyme'] = temp['rhyme'][i]
        result.append(obj)
    return success({
        'song_url': temp['song_url'],
        'song_name': temp['song_name'],
        'lyric': result
    })

@managerApi.route('detail/qa/<uid>/<work_id>/<token>', methods=['GET'])
def qa_check_watching(uid: str, work_id: str, token: str):
    admin = jwt.decode(token, Config.TOKEN_SECRET, algorithms=['HS256'])
    if uid == "":
        return fail("用户信息错误")
    user = User.query.filter_by(id=uid).first()
    if user is None:
        return fail("用户不存在")
    if user.power != 2:
        return fail("权限不足")
    if user.musician == 3:
        query = (db.session.query(
            label("mid", Music.id),
            Music.name, Music.abs_music,
            label("qa", WorkQa.answer),
            label("qa_special", WorkQa.special),
            label("qa_grade", Result.qa_grade),
            label("qa_gradle_count", Result.qa_grade_count)
        ).join(WorkQa, and_(Result.user_from == WorkQa.uid, Result.mid == WorkQa.mid))
                 .join(Music, Result.mid == Music.id)
                 .filter(Result.id == work_id,
                         Result.ex_uid == user.id).first())
        result = {
            'song_url': query.abs_music,
            'song_name': query.name,
        }
    else:
        query = (db.session.query(
            label("mid", Music.id),
            Music.name, Music.abs_music,
            label("qa", WorkQa.answer),
            label("qa_special", WorkQa.special),
            label("pending", WorkQa.pending_reason),
            label("qa_grade", Result.qa_grade),
            label("qa_gradle_count", Result.qa_grade_count)
        ).join(Music, WorkQa.mid == Music.id)
                 .join(Result, and_(Result.mid == WorkQa.mid,
                                     Result.user_from == WorkQa.uid,
                                     Result.type == 2), isouter=True)
                 .filter(WorkQa.id == work_id).first())
        result = {
            'song_url': query.abs_music,
            'song_name': query.name,
        }
        if query.pending != '':
            result['pending'] = query.pending
            return success(result)
    if query.qa_grade is None:
        gradle = []
    else:
        gradle = query.qa_grade.split(',')
    if user.musician != 3:
        current_musician = user.musician
    else:
        current_musician = query.qa_special
    if len(gradle) == 0:
        result['qa'] = format_qa_info(current_musician, query.qa, [], -1)
    else:
        result['qa'] = format_qa_info(current_musician, query.qa, gradle, query.qa_gradle_count)
    if query.qa_gradle_count is not None:
        result["gradle_total"] = query.qa_gradle_count
    return success(result)

def format_qa_info(musician: int, answer: str, qa_grade: [], qa_gradle_count: int):
    qa_query_list = (Question.query
                     .filter_by(musician=musician)
                     .where(Question.sort > 0)
                     .order_by(Question.sort)
                     .all())
    format_answer = json.loads(answer)
    gradle_index = 0
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
            if qa_gradle_count != -1:
                obj["gradle"] = int(qa_grade[gradle_index])
                gradle_index += 1
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
    return result