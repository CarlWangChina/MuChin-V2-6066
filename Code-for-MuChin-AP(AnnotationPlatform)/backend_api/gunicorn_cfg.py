from gunicorn.app.base import BaseApplication
from dotenv import load_dotenv

class Application(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
    def load_config(self):
        config = self.cfg
        config.set('bind', "0.0.0.0:8080")
        config.set('timeout', 300)
        config.set('workers', 4)
        config.set('accesslog', "/www/music-annotation-api/log/gunicorn_access.log")
        config.set('errorlog', "/www/music-annotation-api/log/gunicorn_error.log")
        config.set('access_log_format', '%(t)s %(h)s "%(r)s" %(s)s %(b)s %(D)s "%(a)s"')
    def load(self):
        from app import application
        return application

if __name__ == '__main__':
    load_dotenv(verbose=True, override=True)
    Application(None).run()