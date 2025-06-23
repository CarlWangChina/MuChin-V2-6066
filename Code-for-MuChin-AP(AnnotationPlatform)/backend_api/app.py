from flask_cors import CORS
import CORS
from config import Config
from Code_for_MuChin_AP.backend_api.mapy.__init__ import create_app

app = create_app()
CORS(app)
application = app

if __name__ == '__main__':
    app.run(threaded=True, debug=Config.DEBUG)