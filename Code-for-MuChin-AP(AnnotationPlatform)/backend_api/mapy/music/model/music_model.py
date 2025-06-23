from mapy import db

class Music(db.Model):
    __tablename__ = 'music_info'
    id = db.Column(db.String(32), primary_key=True)
    name = db.Column('name', db.String(255), nullable=False)
    lyric = db.Column('lyric', db.Text)
    abs_music = db.Column('abs_music', db.String(255), nullable=False)
    full_music = db.Column('full_music', db.String(255), nullable=False)
    by = db.Column('from', db.String(255), default='')
    private_id = db.Column('private_id', db.String(255), nullable=False)
    impress_id = db.Column('impress_id', db.Integer, default=0)
    hash_flag = db.Column('task_send_hash', db.String(32), default='')