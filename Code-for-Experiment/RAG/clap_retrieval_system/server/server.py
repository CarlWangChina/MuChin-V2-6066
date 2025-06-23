import os
os.environ['TRANSFORMERS_CACHE'] = '/export/data/home/dev-contributor/.cache/huggingface/hub'
import plyvel
import hashlib
from bottle import Bottle, route, run, request, response, static_file
import vec_search
import json

allowtable_audio = [
    "cb_collect_full",
]
allowtable_lrc = [
    "cb_collect_lrc_rob",
]
audio_root = "/export/data/datasets-mp3/cb/"
db = plyvel.DB('./user_accounts', create_if_missing=True)

with open('encode_list.json', 'r') as f:
    line = f.readline()
    encode_list = dict()
    while line:
        data = json.loads(line)
        encode_list[data["name"]] = data["path"]
        line = f.readline()

app = Bottle()

def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def check_auth(username, password):
    if len(username) == 0:
        return False
    stored_password = db.get(username.encode('utf-8')).decode('utf-8')
    if stored_password:
        hashed_password = hash_password(password.strip())
        if stored_password == hashed_password:
            return True
    print(username, password, stored_password, hashed_password)
    return False

def authenticate():
    response.status = 401
    response.headers['WWW-Authenticate'] = 'Basic realm="Authentication Required"'
    return 'Unauthorized'

@app.route('/')
def index():
    auth = request.auth
    if not auth or not check_auth(auth[0], auth[1]):
        return authenticate()
    return static_file('index.html', root='webroot')

@app.route('/api/<vectable>', method='POST')
def api_handler(vectable):
    auth = request.auth
    if not auth or not check_auth(auth[0], auth[1]):
        return authenticate()
    try:
        post_data = json.loads(request.body.read().decode('utf-8'))
    except Exception as err:
        print(err)
        return "[]"
    translation = False
    mixSearch = False
    lrcMode = False
    if "translation" in post_data and isinstance(post_data["translation"], bool) and post_data["translation"] == True:
        translation = True
    if "mix" in post_data and isinstance(post_data["mix"], bool) and post_data["mix"] == True:
        mixSearch = True
    if "data" in post_data and isinstance(post_data["data"], str):
        print(post_data["data"])
        res = []
        prompt = []
        if vectable in allowtable_lrc:
            print("lrc mode")
            res, prompt = vec_search.search_data_by_lrc(text_data=post_data["data"].split("\n"), table=vectable, translate=translation, mixSearch=mixSearch)
        elif vectable in allowtable_audio:
            print("audio mode")
            res, prompt = vec_search.search_data_by_audio(text_data=post_data["data"].split("\n"), table=vectable, translate=translation, mixSearch=mixSearch)
        response.content_type = 'text/plain'
        res_path = [[row.path for row in item] for item in res]
        res_name = None
        res_lrc = None
        try:
            res_lrc_tmp = [[row.lrc for row in item] for item in res]
            res_songname_tmp = [[row.songName for row in item] for item in res]
            res_lrc = res_lrc_tmp
            res_name = res_songname_tmp
        except Exception:
            pass
        return json.dumps([res_path, prompt, res_lrc, res_name])
    else:
        return "[]"

@app.route('/file/<name>')
def get_file(name):
    auth = request.auth
    if not auth or not check_auth(auth[0], auth[1]):
        return authenticate()
    if name in encode_list:
        file_path = encode_list[name]
        _, file_extension = os.path.splitext(file_path)
        if file_extension == '.mp3':
            mimetype = 'audio/mpeg'
        elif file_extension == '.flac':
            mimetype = 'audio/flac'
        elif file_extension == '.wav':
            mimetype = 'audio/wav'
        else:
            return "Unsupported file type"
        return static_file(file_path.replace(audio_root, "/"), root=audio_root, mimetype=mimetype)
    else:
        return "File not found"

app.run(host='localhost', port=8090)