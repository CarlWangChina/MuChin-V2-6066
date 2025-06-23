from mapy import db

class AnswerUi(db.Model):
    __tablename__ = 'answer_ui_info'
    id = db.Column(db.String(32), primary_key=True)
    title = db.Column('sub_title', db.String(255), nullable=False)
    qid = db.Column('qid', db.String(32), nullable=False)
    cid = db.Column('cid', db.String(255), nullable=False)
    tips = db.Column('tips', db.String(255), nullable=False)
    type = db.Column('type', db.String(2), nullable=False)
    source = db.Column('source', db.Integer)
    exam_tips = db.Column('exam_tips', db.String(255), nullable=False)
    sort = db.Column('sort', db.Integer)
    ui_min_text_len = db.Column('ui_min_text_len', db.Integer)
    ui_min_tag_num = db.Column('ui_min_tag_num', db.Integer)
    content_need_gpt = db.Column('content_need_gpt', db.Integer)