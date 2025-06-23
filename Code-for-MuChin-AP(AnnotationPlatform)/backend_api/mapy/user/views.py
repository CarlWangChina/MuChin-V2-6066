import json
import jwt
from flask import request
from config import Config
from . import api as apiUser
from ..result.network import success, fail
from Code_for_MuChin_AP_AnnotationPlatform.backend_api.mapy.user.models.user_model import User

@apiUser.route('/login', methods=['POST'])
def user_login():
    data = json.loads(request.data)
    account = data['account']
    pwd = data['password']
    user = User.query.filter_by(name=account, delete=0).first()
    if user is None:
        return fail(msg='该用户不存在')
    else:
        if user.password == pwd:
            tk = jwt.encode({'id': user.id, 'power': user.power, 'musician': user.musician}, Config.TOKEN_SECRET, algorithm='HS256')
            return success(data={'token': tk, 'power': user.power, 'musician': user.musician, 'do': user.can_charging})
        else:
            return fail(msg='账号或密码错误')

def check_user_can_used(token: str):
    try:
        user = jwt.decode(token, Config.TOKEN_SECRET, algorithms=['HS256'])
    except Exception:
        raise Exception("用户信息获取失败，请重新尝试")
    query_user = User.query.filter_by(id=user['id'], delete=0).first()
    if query_user is None:
        raise Exception("用户身份异常，请联系管理员")
    return user