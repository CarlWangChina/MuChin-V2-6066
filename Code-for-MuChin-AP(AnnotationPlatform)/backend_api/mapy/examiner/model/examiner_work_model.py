from sqlalchemy import DateTime
import datetime
from mapy import db

class ExaminerWork(db.Model):
    __tablename__ = 'work_examiner'
    id = db.Column(db.String(32), primary_key=True)
    ex_uid = db.Column('ex_uid', db.String(32), nullable=False, default='')
    mid = db.Column('mid', db.String(32), nullable=False, default='')
    qa_work_id = db.Column('qa_work_id', db.String(32), nullable=False, default='')
    work_id_1 = db.Column('work_id_1', db.String(32), nullable=False, default='')
    work_id_2 = db.Column('work_id_2', db.String(32), nullable=False, default='')
    editable = db.Column('editable', db.Integer, nullable=False, default=0)
    is_finish = db.Column('is_finish', db.Integer, default=0)
    create_time = db.Column('create_time', DateTime, nullable=False)
    get_time = db.Column('get_time', DateTime, nullable=False)
    finish_time = db.Column('finish_time', DateTime, nullable=False)
    pending_reason = db.Column("pending_reason", db.String(255), default='')

class ExaminerQaWork(db.Model):
    __tablename__ = 'work_qa_examiner'
    id = db.Column(db.Integer, primary_key=True)
    work_id = db.Column('work_id', db.String(32), nullable=False, default='')
    mid = db.Column('mid', db.String(32), nullable=False, default='')
    qa_grade = db.Column('qa_grade', db.String(255), default='', nullable=False)
    qa_grade_count = db.Column('qa_grade_count', db.Integer, default=0)
    ex_uid = db.Column('ex_uid', db.String(32), default='')
    special = db.Column("special", db.Integer, default=0)
    is_finish = db.Column('is_finish', db.Integer, default=0)
    create_time = db.Column('create_time', DateTime, nullable=False)
    get_time = db.Column('get_time', DateTime, nullable=False)
    finish_time = db.Column('finish_time', DateTime, nullable=False)