from mapy import db

class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.String(32), primary_key=True)
    name = db.Column('name', db.String(255), unique=True, nullable=False)
    password = db.Column('password', db.String(255), unique=True, nullable=False)
    power = db.Column('power', db.Integer, unique=True, default=0)
    delete = db.Column('delete', db.Integer, unique=True, default=0)
    musician = db.Column('musician', db.Integer, unique=True, nullable=False, default=0)
    can_charging = db.Column('can_charging', db.Integer, unique=True, nullable=False, default=0)

    def serialize(self):
        return {
            "id": self.id,
            "name": self.name,
            "password": self.password
        }