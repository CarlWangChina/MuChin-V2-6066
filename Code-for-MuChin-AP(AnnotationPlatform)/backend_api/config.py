import os

class Config:
    def __init__(self):
        pass

    TOKEN_SECRET = os.getenv("TOKEN_SECRET")
    DEBUG = True

class DevelopmentConfig(Config):
    PYPINYIN_NO_PHRASES = True
    MYSQL_HOST = os.getenv("MYSQL_HOST")
    MYSQL_USER = os.getenv("MYSQL_USER")
    MYSQL_PSW = os.getenv("MYSQL_PSW")
    MYSQL_DB = os.getenv("MYSQL_DB")
    SQLALCHEMY_ECHO = True
    SQLALCHEMY_DATABASE_URI = ('mysql+pymysql://' + MYSQL_USER + ':' + MYSQL_PSW + '@' + MYSQL_HOST + '/' + MYSQL_DB)

class ProductionConfig(Config):
    PYPINYIN_NO_PHRASES = True
    MYSQL_HOST = os.getenv("MYSQL_HOST")
    MYSQL_USER = os.getenv("MYSQL_USER")
    MYSQL_PSW = os.getenv("MYSQL_PSW")
    MYSQL_DB = os.getenv("MYSQL_DB")
    SQLALCHEMY_ECHO = False
    SQLALCHEMY_DATABASE_URI = ('mysql+pymysql://' + MYSQL_USER + ':' + MYSQL_PSW + '@' + MYSQL_HOST + '/' + MYSQL_DB)

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}