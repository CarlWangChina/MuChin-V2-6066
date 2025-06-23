from flask import Blueprint

managerA = Blueprint('manager', __name__)

from . import views