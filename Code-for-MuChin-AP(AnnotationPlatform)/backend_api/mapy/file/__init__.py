from flask import Blueprint

fileA = Blueprint('file', __name__)
from . import views