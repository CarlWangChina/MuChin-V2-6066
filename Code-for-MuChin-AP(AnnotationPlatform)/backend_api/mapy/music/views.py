import json
import re
import uuid
from datetime import datetime
from flask import request
from sqlalchemy import func, label, or_
from zhconv import convert
from mapy import db
from . import musicApi
from .model.language_model import Language
from .model.music_model import Music
from .model.work_model import WorkMusic, WorkQa, WorkMusicImpressUser, WorkMusicUserAssign, UserMusicAnswerResult
from ..examiner.model.examiner_work_model import ExaminerWork, ExaminerQaWork
from ..examiner.model.result_model import Result
from ..examiner.views import create_qa_task_to_db
from ..result.network import success, fail
from ..user.models.user_model import User
from ..util.character_util import replace_multiple_spaces_with_single_space, is_simplified_chinese, replace_punctuation_with_space, create_rhyme_result_list_by_position, calcul_rhyme_by_pinyin_or_char, get_pin_form_char
from ..util.other import from_session_map_cn, cn_to_session_map

@musicApi.route('/music/list/<token>', methods=['GET'])
def music_list_query(token: str):
    try:
        user = check_user_can_used(token)
    except Exception as err:
        return fail(str(err))
    if user['musician'] in [3, 4, 30]:
        if user['power'] == 1:
            query = (db.session.query(Music.id, Music.name, label("work_id", ExaminerWork.id), ExaminerWork)
                     .join(ExaminerWork, ExaminerWork.mid == Music.id)
                     .filter(ExaminerWork.ex_uid == user['id'], ExaminerWork.is_finish == 0)
                     .all())
            result = []
            for v in query:
                result.append(create_list_item_value(v, 2))
            return success(data={'song_list': result})
        elif user['power'] == 2:
            query = (db.session.query(Music.id, Music.name, label("work_id", ExaminerQaWork.id), ExaminerQaWork)
                     .join(ExaminerQaWork, ExaminerQaWork.mid == Music.id)
                     .filter(ExaminerQaWork.ex_uid == user['id'], ExaminerQaWork.is_finish == 0)
                     .all())
            result = []
            for v in query:
                result.append(create_list_item_value(v, 1, v.ExaminerQaWork.special))
            return success(data={'song_list': result})
    elif user['power'] == 2:
        query = (db.session.query(Music.id, Music.name, WorkQa)
                 .join(WorkQa, WorkQa.mid == Music.id)
                 .filter(WorkQa.uid == user['id'], WorkQa.is_finish == 0)
                 .all())
        result = []
        for v in query:
            result.append(create_list_item_value(v, 1))
        return success(data={'song_list': result})
    elif user['power'] == 1:
        query = (db.session.query(Music.id, Music.name, WorkMusic)
                 .join(WorkMusic, WorkMusic.mid == Music.id)
                 .filter(WorkMusic.uid == user['id'], WorkMusic.is_finish == 0)
                 .all())
        result = []
        for v in query:
            result.append(create_list_item_value(v, 2))
        data = {'song_list': result}
        return success(data=data)
    else:
        return fail("未知错误")

def create_list_item_value(value, editable, special=-1):
    if hasattr(value, 'work_id'):
        name = value.name
        if special == 0:
            name += '(非专业)'
        elif special == 1:
            name += '(专业)'
        return {
            'work_id': value.work_id,
            'music_id': value.id,
            'music_name': name,
            'editable': editable
        }
    return {
        'music_id': value.id,
        'music_name': value.name,
        'editable': editable
    }

@musicApi.route('/song_lyc/<song_id>/<state>/<token>', methods=['GET'])
def query_music_lyc(song_id: str, state: str, token: str):
    lans = Language.query.all()
    new_lans = []
    for i in lans:
        new_lans.append(i.serialize())
    if state == '0':
        music = Music.query.filter_by(id=song_id).first()
        if music is None:
            return fail("该歌曲不存在,code:0")
        song_list = music.lyric.split('\n')
        return success({
            'name': music.name,
            'music': music.full_music,
            'lrc_list': song_list,
            'lans': new_lans
        })
    elif state == '1':
        data = music_lyc_section(song_id, token)
        data['data']['lans'] = new_lans
        return data
    else:
        return fail(msg='未知错误')

@musicApi.route('/song_lyc/<song_id>/<token>', methods=['POST'])
def save_music_lyc(song_id: str, token: str):
    try:
        user = check_user_can_used(token)
    except Exception as err:
        return fail(str(err))
    data = json.loads(request.data)
    work = WorkMusic.query.filter_by(uid=user['id'], mid=song_id, is_finish=0).first()
    if work is None:
        return fail("歌曲信息获取失败")
    song = data['song'].strip()
    if song == "":
        return fail("歌曲内容不能为空")
    convert_text = convert(song, 'zh-cn')
    replace_text = replace_punctuation_with_space(convert_text).strip()
    for c in replace_text:
        if c == ' ' or c == '\n':
            continue
        if not is_simplified_chinese(c):
            return fail("输入的歌曲中不允许包含非简体中文，如果该歌曲确实有非简体中文的歌词需要输入，请在上方包含语言中填写并确认跳过")
    song_lrc = replace_multiple_spaces_with_single_space(replace_text)
    if song_lrc == "":
        return fail("请检查您的歌词已确保内容不为空")
    if song_lrc == work.song_lrc:
        return success({})
    if work.song_lrc is None:
        ori_song_split = []
    else:
        ori_song_split = work.song_lrc.split('\n')
    new_song_split = song_lrc.split('\n')
    work.song_lrc = song_lrc
    if len(ori_song_split) != len(new_song_split):
        work.rhyme = ''
    db.session.commit()
    return success({})

@musicApi.route('/song_lyc/section/<song_id>/<token>', methods=['GET', 'POST'])
def music_lyc_section(song_id: str, token: str):
    try:
        user = check_user_can_used(token)
    except Exception as err:
        return fail(str(err))
    if request.method == 'GET':
        query = (db.session.query(Music.id, Music.name, Music.full_music, WorkMusic)
                 .join(WorkMusic, WorkMusic.mid == Music.id)
                 .filter(WorkMusic.uid == user['id'],
                         WorkMusic.is_finish == 0,
                         WorkMusic.mid == song_id)
                 .first())
        if query is None:
            return fail("该歌曲不存在,code:1")
        new = query.WorkMusic.song_lrc.split('\n')
        section = []
        if query.WorkMusic.section is None or (len(query.WorkMusic.section) == 0):
            for i in new:
                section.append({'words': i})
        else:
            for i, v in enumerate(new):
                sub_sec = query.WorkMusic.section[i * 2:i * 2 + 2]
                if sub_sec is None or sub_sec == '':
                    section.append({'words': v})
                else:
                    sub_type = sub_sec[0]
                    sub_num = sub_sec[1]
                    section.append({
                        'words': v,
                        'type': (from_session_map_cn(sub_type) + str(ord(sub_num) - 96))
                    })
        lans = Language.query.all()
        new_lans = []
        for i in lans:
            new_lans.append(i.serialize())
        return success(data={
            'name': query.name,
            'music': query.full_music,
            'section_list': section,
            'lans': new_lans
        })
    elif request.method == 'POST':
        work = WorkMusic.query.filter_by(uid=user['id'], mid=song_id, is_finish=0).first()
        if work is None:
            return fail("歌曲信息获取错误，请重新尝试,code:1")
        data = json.loads(request.data)
        song_lrc = work.song_lrc
        if song_lrc == '':
            return fail("歌曲信息获取错误，请返回上一步重新尝试,code:1")
        song_count = len(song_lrc.split('\n'))
        format_data = ''
        for i in data:
            sub_type = i[0:len(i) - 1]
            sub_num = i[-1:]
            format_data += cn_to_session_map(sub_type) + chr(int(sub_num) + 96)
        if len(format_data) != (song_count * 2):
            return fail("段落信息保存失败，请重新尝试")
        work.section = format_data
        db.session.commit()
        return success({})

@musicApi.route('/song_lyc/rhyme/<song_id>/<token>', methods=['GET', 'POST'])
def music_lyc_rhyme(song_id: str, token: str):
    try:
        user = check_user_can_used(token)
    except Exception as err:
        return fail(str(err))
    if request.method == 'GET':
        query = (db.session.query(Music.id, Music.name, Music.full_music, WorkMusic)
                 .join(WorkMusic, WorkMusic.mid == Music.id)
                 .filter(WorkMusic.uid == user['id'],
                         WorkMusic.is_finish == 0,
                         WorkMusic.mid == song_id)
                 .first())
        if query is None:
            return fail("该歌曲不存在,code:2")
        lrc_list = query.WorkMusic.song_lrc.split('\n')
        result = []
        if user['musician'] == 3 and len(query.WorkMusic.rhyme) > 0:
            for i, v in enumerate(lrc_list):
                cur_section = query.WorkMusic.section[i * 2: i * 2 + 2]
                obj = {
                    'line': i,
                    'type': from_session_map_cn(cur_section[0]) + str(ord(cur_section[1]) - 96),
                    'words': v
                }
                if query.WorkMusic.rhyme[i] != 'Z':
                    obj['rhyme'] = query.WorkMusic.rhyme[i]
                result.append(obj)
        else:
            cur_section = query.WorkMusic.section[0:2]
            start = 0
            end = 0
            for i, v in enumerate(lrc_list):
                sub_sec = query.WorkMusic.section[i * 2: i * 2 + 2]
                if sub_sec == cur_section:
                    end = i
                else:
                    result.extend(create_rhyme_result_list_by_position(lrc_list, cur_section, start, end))
                    cur_section = sub_sec
                    start = i
                    end = i
            result.extend(create_rhyme_result_list_by_position(lrc_list, cur_section, start, end))
        lans = Language.query.all()
        new_lans = []
        for i in lans:
            new_lans.append(i.serialize())
        return success(data={
            'name': query.name,
            'music': query.full_music,
            'rhyme_list': result,
            'lans': new_lans
        })
    elif request.method == 'POST':
        work = WorkMusic.query.filter_by(uid=user['id'], mid=song_id, is_finish=0).first()
        if work is None:
            return fail("歌曲信息获取错误，请重新尝试,code:2")
        data = json.loads(request.data)
        song_lrc = work.song_lrc
        if song_lrc == '':
            return fail("歌曲信息获取错误，请返回上一步重新尝试，code:2")
        song_count = len(song_lrc.split('\n'))
        if len(data) == 0 or song_count != len(data):
            return fail("数据异常，请重新尝试")
        rhyme = ""
        for i in data:
            if i == '':
                rhyme += 'Z'
            else:
                rhyme += i
        if rhyme == "":
            return fail("押韵存储错误，请重新尝试")
        work.rhyme = rhyme
        work.is_finish = 1
        work.finish_time = datetime.now()
        db.session.commit()
        try:
            check_music_is_same_and_transmit(user, song_id, work)
        except Exception as err:
            return fail(str(err))
        return success({})

def check_music_is_same_and_transmit(user, song_id, current_work):
    other_work = WorkMusic.query.filter(WorkMusic.uid != user['id'], WorkMusic.mid == song_id,
                                        WorkMusic.is_finish == 1).first()
    if other_work is not None:
        is_same = set(other_work.extra_lan.split(',')) == set(current_work.extra_lan.split(','))
        if current_work.section is None:
            my_section: str = ''
        else:
            my_section: str = re.sub(r'[a-zA-Z]', '', current_work.section)
        if other_work.section is None:
            other_section = ''
        else:
            other_section = re.sub(r'[a-zA-Z]', '', other_work.section)
        if (other_work.extra_lan is not None and
                other_work.extra_lan != "" and
                current_work.extra_lan is not None and
                current_work.extra_lan != ""
                and is_same):
            handle_same_skip_music(user, song_id, current_work, other_work)
        elif (other_work.rhyme == current_work.rhyme and
              len(other_work.rhyme) > 0 and
              len(current_work.rhyme) > 0 and
              other_section == my_section and
              len(other_section) > 0 and
              len(my_section) > 0 and
              other_work.song_lrc == current_work.song_lrc and
              len(other_work.song_lrc) > 0 and
              len(current_work.song_lrc) > 0 and
              (other_work.extra_lan is None or other_work.extra_lan == "") and
              (current_work.extra_lan is None or current_work.extra_lan == "")):
            handle_same_annotation_music(user, song_id, current_work, other_work)
        else:
            handle_diff_annotation_music(user, song_id, current_work, other_work)
    else:
        if user['musician'] == 2:
            if len(current_work.extra_lan) == 0:
                lrc_extra_lan_state = 1
            create_qa_task_to_db(song_id, current_work.song_lrc)
            else:
                lrc_extra_lan_state = 2
        db.session.add(Result(
            id=str(uuid.uuid4()).replace('-', ''),
            mid=song_id,
            type=1,
            lrc_song=current_work.song_lrc,
            lrc_section=current_work.section,
            lrc_rhyme=current_work.rhyme,
            lrc_extra_lan=current_work.extra_lan,
            user_from=user['id'],
            time=datetime.now(),
            lrc_extra_lan_state=lrc_extra_lan_state
        ))
        db.session.add(UserMusicAnswerResult(
            uid=current_work.uid,
            mid=song_id,
            state=6,
            create_time=datetime.now()
        ))
        db.session.commit()

def handle_same_skip_music(user, song_id: str, current_work: WorkMusic, other_work: WorkMusic):
    special_user = find_special_annotator()
    result = Result.query.filter_by(mid=song_id, type=1).first()
    if other_work.uid == special_user.id and result is not None:
        result.user_from = f'{result.user_from},{user["id"]}'
        db.session.add(
            UserMusicAnswerResult(
                uid=user['id'],
                mid=song_id,
                state=1,
                create_time=datetime.now()
            )
        )
        db.session.commit()
        return
    if result is not None:
        raise Exception("请勿重复提交数据 ,code: 01 ")
    check_result = Result(
        id=str(uuid.uuid4()).replace('-', ''),
        mid=song_id,
        type=1,
        lrc_extra_lan=other_work.extra_lan,
        user_from=other_work.uid + ',' + current_work.uid,
        time=datetime.now(),
        lrc_extra_lan_state=2
    )
    db.session.add(check_result)
    special_user = find_special_annotator()
    if other_work.uid == special_user.id:
        other_state = 6
    else:
        other_state = 1
    db.session.add(UserMusicAnswerResult(
        uid=other_work.uid,
        mid=song_id,
        state=other_state,
        create_time=datetime.now()
    ))
    if current_work.uid == special_user.id:
        current_state = 6
    else:
        current_state = 1
    db.session.add(UserMusicAnswerResult(
        uid=current_work.uid,
        mid=song_id,
        state=current_state,
        create_time=datetime.now()
    ))
    db.session.commit()

def handle_same_annotation_music(user, song_id: str, current_work: WorkMusic, other_work: WorkMusic):
    special_user = find_special_annotator()
    result = Result.query.filter_by(mid=song_id, type=1).first()
    if other_work.uid == special_user.id and result is not None:
        result.user_from = f'{result.user_from},{user["id"]}'
        result.lrc_section = f'{result.lrc_section},{current_work.section}'
        db.session.add(
            UserMusicAnswerResult(
                uid=user['id'],
                mid=song_id,
                state=1,
                create_time=datetime.now()
            )
        )
        db.session.commit()
        return
    if result is not None:
        raise Exception("请勿重复提交数据 ,code: 02 ")
    if other_work.section == current_work.section:
        save_section = other_work.section
    else:
        save_section = other_work.section + ',' + current_work.section
    check_result = Result(
        id=str(uuid.uuid4()).replace('-', ''),
        mid=song_id,
        type=1,
        lrc_song=other_work.song_lrc,
        lrc_section=save_section,
        lrc_rhyme=other_work.rhyme,
        user_from=other_work.uid + ',' + current_work.uid,
        time=datetime.now(),
        lrc_extra_lan_state=1
    )
    create_qa_task_to_db(song_id, other_work.song_lrc)
    db.session.add(check_result)
    special_user = find_special_annotator()
    if other_work.uid == special_user.id:
        other_state = 6
    else:
        other_state = 1
    db.session.add(UserMusicAnswerResult(
        uid=other_work.uid,
        mid=song_id,
        state=other_state,
        create_time=datetime.now()
    ))
    if current_work.uid == special_user.id:
        current_state = 6
    else:
        current_state = 1
    db.session.add(UserMusicAnswerResult(
        uid=current_work.uid,
        mid=song_id,
        state=current_state,
        create_time=datetime.now()
    ))
    db.session.commit()

def handle_diff_annotation_music(user, song_id: str, current_work: WorkMusic, other_work: WorkMusic):
    special_user = find_special_annotator()
    result = Result.query.filter_by(mid=song_id, type=1).first()
    if result is None and special_user.id == user['id']:
        if current_work.extra_lan == '':
            lrc_extra_lan_state = 1
        create_qa_task_to_db(song_id, current_work.song_lrc)
        else:
            lrc_extra_lan_state = 2
        check_result = Result(
            id=str(uuid.uuid4()).replace('-', ''),
            mid=song_id,
            type=1,
            lrc_song=current_work.song_lrc,
            lrc_section=current_work.section,
            lrc_rhyme=current_work.rhyme,
            lrc_extra_lan=current_work.extra_lan,
            user_from=user['id'],
            time=datetime.now(),
            lrc_extra_lan_state=lrc_extra_lan_state
        )
        db.session.add(check_result)
        answer_result1 = UserMusicAnswerResult(
            uid=user['id'],
            mid=song_id,
            state=6,
            create_time=datetime.now()
        )
        answer_result2 = UserMusicAnswerResult(
            uid=other_work.uid,
            mid=song_id,
            state=3,
            create_time=datetime.now()
        )
        db.session.add(answer_result1)
        db.session.add(answer_result2)
        db.session.commit()
    elif result is not None and special_user.id != user['id']:
        db.session.add(
            UserMusicAnswerResult(
                uid=user['id'],
                mid=song_id,
                state=3,
                create_time=datetime.now()
            )
        )
        db.session.commit()
    else:
        examiner_work = ExaminerWork.query.filter_by(mid=song_id, editable=2).first()
        if examiner_work is None:
            new_work = ExaminerWork(
                id=str(uuid.uuid4()).replace('-', ''),
                mid=song_id,
                editable=2,
                work_id_1=current_work.id,
                work_id_2=other_work.id,
                create_time=datetime.now()
            )
            db.session.add(new_work)
            db.session.commit()

@musicApi.route('/song_lyc/rhyme/calcul/<token>', methods=['POST'])
def calcul_rhyme_by_pinyin(token: str):
    try:
        user = check_user_can_used(token)
    except Exception as err:
        return fail(str(err))
    data = json.loads(request.data)
    return success(data=calcul_rhyme_by_pinyin_or_char(data))

@musicApi.route('/song/lan/check/<submit_type>/<song_id>/<token>', methods=['POST'])
def song_lan_check(submit_type: str, song_id: str, token: str):
    try:
        user = check_user_can_used(token)
    except Exception as err:
        return fail(str(err))
    lan = json.loads(request.data)
    if len(lan) == 0:
        return fail("未知错误")
    work = WorkMusic.query.filter(WorkMusic.uid == user['id'],
                                  WorkMusic.mid == song_id,
                                  or_(WorkMusic.is_finish == 0, WorkMusic.is_finish == 2)).first()
    if work is None:
        return fail("歌曲信息获取错误，请重新尝试,code:4")
    lan_str = ""
    for i in lan:
        lan_str += i + ","
    work.extra_lan = lan_str[0:len(lan_str) - 1]
    if user['musician'] in [3, 4]:
        work.is_finish = 1
        work.finish_time = datetime.now()
        ex_work = ExaminerWork.query.filter_by(mid=song_id, ex_uid=user['id'], editable=2, is_finish=0).first()
        if ex_work is None:
            return fail("请勿重复提交，code:01")
        ex_work.is_finish = 1
        ex_work.finish_time = datetime.now()
        query_result = Result.query.filter_by(mid=song_id, type=1).first()
        if query_result is None:
            new = Result(
                id=str(uuid.uuid4()).replace('-', ''),
                mid=song_id,
                type=1,
                lrc_extra_lan=lan_str[0:len(lan_str) - 1],
                user_from=user['id'],
                time=datetime.now(),
                lrc_extra_lan_state=2,
                ex_uid=user['id']
            )
            db.session.add(new)
        else:
            return fail("数据提交失败 ,code 0")
        users = db.session.query(WorkMusic.uid).filter(WorkMusic.mid == song_id, WorkMusic.uid != user['id']).all()
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
        return success({})
    else:
        work.is_finish = 1
        work.finish_time = datetime.now()
        db.session.commit()
        try:
            check_music_is_same_and_transmit(user, song_id, work)
        except Exception as err:
            return fail(str(err))
        return success({})

@musicApi.route('/refresh/new/<token>', methods=['GET'])
def task_get_new_work(token: str):
    try:
        user = check_user_can_used(token)
    except Exception as err:
        return fail(str(err))
    limit = 20
    if user['musician'] in [3, 4]:
        return fail("错误请求")
    elif user['musician'] == 2:
        return fail("无需领取")
    elif user['power'] == 1:
        count_query = (db.session.query(label("count", func.count(WorkMusic.uid)))
                       .filter(WorkMusic.uid == user['id'], WorkMusic.is_finish == 0).first())
        need_add_count = limit - count_query.count
        if need_add_count <= 0:
            return success({})
        task_random_id = str(uuid.uuid4()).replace('-', '')
        db.session.execute(db.text(f"UPDATE music SET hash_flag='{task_random_id}' WHERE id IN (SELECT id FROM music ORDER BY RAND() LIMIT {need_add_count})"))
        db.session.commit()
        new_music_list = Music.query.filter_by(hash_flag=task_random_id).all()
        new_work_list = []
        new_user_assign_list = []
        result = []
        for v in new_music_list:
            result.append(create_list_item_value(v, 2))
        new_work_list.append(
            WorkMusic(
                mid=v.id,
                uid=user['id'],
                get_time=datetime.now()
            )
        )
        query_assign = WorkMusicUserAssign.query.filter_by(mid=v.id).first()
        if query_assign is None:
            new_user_assign_list.append(
                WorkMusicUserAssign(
                    mid=v.id,
                    uid1=user['id'],
                    create_time=datetime.now()
                )
            )
        else:
            query_assign.uid2 = user['id']
            query_assign.update_time = datetime.now()
        query_impress = WorkMusicImpressUser.query.filter_by(impress_id=v.impress_id, uid=user['id']).first()
        if query_impress is None:
            db.session.add(WorkMusicImpressUser(
                impress_id=v.impress_id,
                count=1,
                uid=user['id']
            ))
            db.session.commit()
        else:
            query_impress.count += 1
            db.session.commit()
        db.session.add_all(new_work_list)
        db.session.add_all(new_user_assign_list)
        db.session.commit()
        for song in new_music_list:
            song.hash_flag = ''
        db.session.commit()
        return success(data={'song_list': result})
    elif user['power'] == 2:
        count_query = (db.session.query(label("count", func.count(WorkQa.uid)))
                       .filter(WorkQa.uid == user['id'], WorkQa.is_finish == 0).first())
        need_add_count = limit - count_query.count
        if need_add_count <= 0:
            return success({})
        sql_str = f"UPDATE music SET hash_flag='{str(uuid.uuid4()).replace('-', '')}' WHERE id IN (SELECT id FROM music ORDER BY RAND() LIMIT {need_add_count})"
        db.session.execute(db.text(sql_str))
        db.session.commit()
        query = (db.session.query(Music.id, Music.name, WorkQa)
                 .join(WorkQa, WorkQa.mid == Music.id)
                 .filter(WorkQa.uid == user['id'], WorkQa.is_finish == 0)
                 .all())
        result = []
        for v in query:
            result.append(create_list_item_value(v, 1))
        return success(data={'song_list': result})

@musicApi.route("/pending/<song_id>/<token>", methods=['POST'])
def pending_music(song_id: str, token: str):
    try:
        user = check_user_can_used(token)
    except Exception as err:
        return fail(str(err))
    if user['musician'] == 2:
        return fail("该账号不允许挂起")
    reason_obj = json.loads(request.data)
    music = WorkMusic.query.filter_by(mid=song_id, uid=user['id'], is_finish=0).first()
    if music is not None:
        music.is_finish = 2
        if '