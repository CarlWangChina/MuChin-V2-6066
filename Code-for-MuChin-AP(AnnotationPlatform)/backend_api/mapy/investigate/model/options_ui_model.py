from mapy import db

class OptionUi(db.Model):
    __tablename__ = 'answer_ui_option'
    id = db.Column(db.String(32), primary_key=True)
    option = db.Column('option', db.String(255), nullable=False)
    aid = db.Column('aid', db.String(32), nullable=False)
    must = db.Column('must', db.String(1), nullable=False)
    ui_min_text_len = db.Column('ui_min_text_len', db.Integer)

    def format(self, add_must=False):
        obj = {
            'id': self.id,
            'name': self.option,
            'min_len': self.ui_min_text_len
        }

        if add_must:
            obj['must'] = self.must
        return obj