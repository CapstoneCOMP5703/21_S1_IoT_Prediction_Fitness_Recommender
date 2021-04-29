from flask import Flask
from flask import request, render_template, redirect, url_for, session, g
# from flask_login import LoginManager, current_user, login_user, login_required, logout_user, UserMixin
from dataclasses import dataclass
from datetime import timedelta

#调用自己写的model.py
from models import UserModel,db

app= Flask(__name__,static_url_path="/")
app.config['SECRET_KEY'] = "sdfklas5fa2k42j"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///<db_name>.db' #这里链接数据库
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

#调用自己写的auth.py
from auth import login

db.init_app(app)
login.init_app(app)

@app.before_first_request
def create_all():
    db.create_all()

# class User(UserMixin):
#     pass

# users = [
#     {'id':'1', 'username': 'Admin', 'password': '123456'},
#     {'id':'2', 'username': 'Eason', 'password': '888888'},
#     {'id':'3', 'username': 'Tommy', 'password': '666666'}
# ]

# def query_user(user_name):
#     for user in users:
#         if user_name == user['username']:
#             return user

# login_manager = LoginManager()
# login_manager.login_view = 'login'
# login_manager.login_message_category = 'info'
# login_manager.login_message = 'Access denied.'
# login_manager.init_app(app)

# @login.user_loader
# def load_user(id):
#     return UserModel.query.get(int(id))

#路由主页
@app.route("/")
def homepage():           
    return render_template("homepage.html")

#路由运动推荐，目前需要登录才能使用
@app.route("/workoutrec")
def workoutRec():       
    if not g.user:
        return redirect(url_for('login'))
    return render_template("workoutrec.html")

#路由饮食推荐
@app.route("/dietrec")
def mealRec():          
    return render_template("dietrec.html")

#路由运动记录    
@app.route("/activitylog")
def activitylog():        
    return render_template("activitylog.html")

#路由用户登录后显示的页面
@app.route("/profile")
def profile():        
    return render_template("profile.html")

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         user_name = request.form.get('username')
#         user = query_user(user_name)
#         if user is not None and request.form['password'] == user['password']:

#             curr_user = User()
#             curr_user.username = user_name

#             # 通过Flask-Login的login_user方法登录用户
#             login_user(curr_user)

#             return redirect(url_for('workoutRec'))

#         flash('Wrong username or password!')

#     # GET 请求
#     return render_template('login.html')

# @app.route('/logout')
# # @login_required
# def logout():
#     logout_user()
#     return redirect(url_for('homepage'))

#用户登录
@app.route('/login', methods = ['POST', 'GET'])
def login():
    if current_user.is_authenticated:
        return redirect('/profile')
     
    if request.method == 'POST':
        username = request.form['username']
        user = UserModel.query.filter_by(username = username).first()
        if user is not None and user.check_password(request.form['password']):
            login_user(user)
            return redirect('/profile')
     
    return render_template('sign.html')

#用户注册-目前注释掉了，准备先试登录
# @app.route('/register', methods=['POST', 'GET'])
# def register():
#     if current_user.is_authenticated:
#         return redirect('/profile')
     
#     if request.method == 'POST':
#         # email = request.form['email']
#         username = request.form['username']
#         password = request.form['password']
 
#         if UserModel.query.filter_by(email=email):
#             return ('Email already Present')
             
#         user = UserModel(email=email, username=username)
#         user.set_password(password)
#         db.session.add(user)
#         db.session.commit()
#         return redirect('/login')
#     return render_template('register.html')

#用户登出
@app.route('/logout')
def logout():
    logout_user()
    return redirect('/homepage')