from flask import Blueprint

import p
import r
import s

Api = Blueprint('examiner', __name__)
from . import views