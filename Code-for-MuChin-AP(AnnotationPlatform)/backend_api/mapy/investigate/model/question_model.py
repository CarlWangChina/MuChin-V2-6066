from mapy import db

class Question(db.Model):
    __tablename__ = 'question_info'
    id = db.Column(db.String(32), primary_key=True)
    question = db.Column('question', db.String(255), nullable=False)
    musician = db.Column('musician', db.String(1), nullable=False)
    sort = db.Column('sort', db.Integer)