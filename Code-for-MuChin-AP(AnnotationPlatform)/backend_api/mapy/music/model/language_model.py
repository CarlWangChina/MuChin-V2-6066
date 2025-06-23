from mapy import db

class Language(db.Model):
    __tablename__ = 'music_language'
    id = db.Column('id', db.String(255), primary_key=True)
    lan = db.Column('lan', db.String(255), nullable=False)

    def serialize(self):
        return {
            'value': self.id,
            'label': self.lan
        }