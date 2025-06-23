from sqlalchemy import DateTime
import datetime
from mapy import db

class Result(db.Model):
    __tablename__ = 'result'
    id = db.Column(db.String(32), primary_key=True)
    mid = db.Column('mid', db.String(255), nullable=False)
    type = db.Column("type", db.Integer, default=1)
    user_from = db.Column('user_from', db.String(255), default='')
    time = db.Column("time", DateTime, nullable=False)
    lrc_song = db.Column('lrc_song', db.Text, default='')
    lrc_section = db.Column('lrc_section', db.String(800), default='')
    lrc_rhyme = db.Column('lrc_rhyme', db.String(255), default='')
    lrc_extra_lan = db.Column('lrc_extra_lan', db.String(255), default='')
    lrc_extra_lan_state = db.Column('lrc_extra_lan_state', db.Integer, default=0)
    qa_grade = db.Column('qa_grade', db.String(255), default='', nullable=False)
    qa_grade_count = db.Column('qa_grade_count', db.Integer, default=0)
    ex_uid = db.Column('ex_uid', db.String(32), default='')

class Result2(db.Model):
    __tablename__ = 'result_temp'
    id = db.Column(db.String(32), primary_key=True)
    mid = db.Column('mid', db.String(255), nullable=False)
    type = db.Column("type", db.Integer, default=1)
    user_from = db.Column('user_from', db.String(255), default='')
    time = db.Column("time", DateTime, nullable=False)
    lrc_song = db.Column('lrc_song', db.Text, default='')
    lrc_section = db.Column('lrc_section', db.String(255), default='')
    lrc_rhyme = db.Column('lrc_rhyme', db.String(255), default='')
    lrc_extra_lan = db.Column('lrc_extra_lan', db.String(255), default='')
    qa_grade = db.Column('qa_grade', db.String(255), default='', nullable=False)
    qa_grade_count = db.Column('qa_grade_count', db.Integer, default=0)