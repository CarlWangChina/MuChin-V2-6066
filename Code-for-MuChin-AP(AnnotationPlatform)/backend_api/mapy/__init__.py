import os
from dotenv import load_dotenv
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import config

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    load_dotenv(verbose=True, override=True)
    config_name = os.getenv('OS_ENVIRONMENT')
    app.config.from_object(config[config_name])

    db.init_app(app)
    from .user import api as apiUser
    app.register_blueprint(apiUser, url_prefix='/user')
    from .music import api as musicApi
    app.register_blueprint(musicApi, url_prefix='/music')
    from .file import api as fileApi
    app.register_blueprint(fileApi, url_prefix='/file')
    from .investigate import api as qaApi
    app.register_blueprint(qaApi, url_prefix='/investigate')
    from .examiner import api as examinerApi
    app.register_blueprint(examinerApi, url_prefix='/examiner')
    from .manager import api as managerApi
    app.register_blueprint(managerApi, url_prefix='/manager')
    return app