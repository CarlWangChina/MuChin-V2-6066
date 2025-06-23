from enum import Enum

class State(Enum):
    __abstract__ = True

class NetWorkState(State):
    SUCCESS = 200
    FAIL = 400

def result(data=None, code: State = NetWorkState.SUCCESS, msg: str = ""):
    if data is None:
        data = {}
    return {
        "code": code.value,
        "data": data,
        "msg": msg
    }

def success(data: {} = {}):
    return result(data=data, code=NetWorkState.SUCCESS)

def fail(msg: str):
    return result(code=NetWorkState.FAIL, msg=msg)