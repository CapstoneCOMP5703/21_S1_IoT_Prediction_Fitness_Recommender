from flask_login import LoginManager
from models import UserModel

login = LoginManager()
 
@login.user_loader
def load_user(id):
    return UserModel.query.get(int(id))