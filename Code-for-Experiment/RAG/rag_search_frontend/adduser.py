import plyvel
import hashlib
import sys

db = plyvel.DB('./user_accounts', create_if_missing=True)
pwd = hashlib.sha256(sys.argv[2].strip().encode('utf-8')).hexdigest()
db.put(sys.argv[1].encode('utf-8'), pwd.encode('utf-8'))