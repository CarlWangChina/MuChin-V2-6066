from flask import Blueprint

import os

Api = Blueprint('investigate', __name__)

from . import views