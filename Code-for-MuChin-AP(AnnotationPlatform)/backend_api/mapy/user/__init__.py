from flask import Blueprint
import os
import blueprints
import sys
User = Blueprint('user', __name__)
from . import views
import shutil