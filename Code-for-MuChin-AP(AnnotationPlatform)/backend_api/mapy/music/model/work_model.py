from sqlalchemy import DateTime
import datetime
from mapy import db

class WorkMusic(db.Model):
    __tablename__ = 'work_4_music'
    id = db.Column(db.Integer, primary_key=True)
    uid = db.Column('uid', db.String(32), nullable=False, default='')
    mid = db.Column('mid', db.String(32), nullable=False, default='')
    song_lrc = db.Column('song_lrc', db.Text, default='')
    section = db.Column('section', db.String(400), default='')
    rhyme = db.Column('rhyme', db.String(255), default='')
    extra_lan = db.Column('extra_lan', db.String(255), default='')
    is_finish = db.Column('is_finish', db.Integer, default=0)
    pending_reason = db.Column("pending_reason", db.String(255), default='')
    get_time = db.Column("get_time", DateTime)
    finish_time = db.Column("finish_time", DateTime)

class WorkMusicTemp(db.Model):
    __tablename__ = 'work_4_music_temp'
    id = db.Column(db.Integer, primary_key=True)
    uid = db.Column('uid', db.String(32), nullable=False, default='')
    mid = db.Column('mid', db.String(32), nullable=False, default='')
    song_lrc = db.Column('song_lrc', db.Text, default='')
    section = db.Column('section', db.String(400), default='')
    rhyme = db.Column('rhyme', db.String(255), default='')
    extra_lan = db.Column('extra_lan', db.String(255), default='')
    is_finish = db.Column('is_finish', db.Integer, default=0)
    get_time = db.Column("get_time", DateTime)
    finish_time = db.Column("finish_time", DateTime)
    pending_reason = db.Column("pending_reason", db.String(255), default='')

class WorkMusicUserAssign(db.Model):
    __tablename__ = 'work_4_music_user_assign'
    id = db.Column('work_id', db.Integer, primary_key=True)
    mid = db.Column('music_id', db.String(32), nullable=False, default='')
    uid1 = db.Column('user_id_1', db.String(32), nullable=False, default='')
    uid2 = db.Column('user_id_2', db.String(32), nullable=False, default='')
    create_time = db.Column('create_time', DateTime, nullable=False)
    update_time = db.Column('update_time', DateTime, nullable=False)

class WorkMusicImpressUser(db.Model):
    __tablename__ = 'work_4_music_impress_user'
    id = db.Column('work_id', db.Integer, primary_key=True)
    impress_id = db.Column('impress_id', db.Integer, nullable=False)
    count = db.Column('impress_count', db.Integer, nullable=False)
    uid = db.Column('user_id', db.String(32), nullable=False, default='')

class WorkQa(db.Model):
    __tablename__ = 'work_4_qa'
    id = db.Column(db.Integer, primary_key=True)
    uid = db.Column('uid', db.String(32), default='')
    mid = db.Column('mid', db.String(32), nullable=False, default='')
    answer = db.Column('answer', db.Text, default='')
    is_finish = db.Column('is_finish', db.Integer, default=0)
    special = db.Column("special", db.Integer, default=0)
    create_time = db.Column('create_time', DateTime, nullable=False)
    get_time = db.Column('get_time', DateTime, nullable=False)
    finish_time = db.Column('finish_time', DateTime, nullable=False)
    pending_reason = db.Column("pending_reason", db.String(255), default='')
    gpt_sum_content = db.Column('gpt_sum_content', db.String(300), default='')

class UserMusicAnswerResult(db.Model):
    an_id = db.Column(db.Integer, primary_key=True)
    uid = db.Column('user_id', db.String(32), default='')
    mid = db.Column('music_id', db.String(32), default='')
    state = db.Column('answer_state', db.Integer, default=0)
    create_time = db.Column('create_time', DateTime, nullable=False)
    __table_args__ = (
        db.UniqueConstraint('user_id', 'music_id', name='pid'),
    )

class DropUserMusicAnswerResult(db.Model):
    __tablename__ = 'user_music_answer_result_drop_data'
    an_id = db.Column(db.Integer, primary_key=True)
    uid = db.Column('user_id', db.String(32), default='')
    mid = db.Column('music_id', db.String(32), default='')
    state = db.Column('answer_state', db.Integer, default=0)
    create_time = db.Column('create_time', DateTime, nullable=False)