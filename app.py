from flask import Flask
from flask import request, render_template, redirect, url_for, session, g
from dataclasses import dataclass

app= Flask(__name__,static_url_path="/")
app.config['SECRET_KEY'] = "sdfklas5fa2k42j"

@dataclass
class User:
    id: int
    username: str
    password: str

users = [
	User(1, "Admin", "123456"),
	User(2, "Eason", "888888"),
	User(3, "Tommy", "666666"),
]

@app.before_request
def before_request():
    g.user = None
    if 'user_id' in session:
        user = [u for u in users if u.id == session['user_id']][0] #todo 替换成数据库
        g.user = user

@app.route("/")
def homepage():           
    return render_template("homepage.html")

@app.route("/workrec")
def workoutRec():           
    return render_template("workoutRec.html")

@app.route("/dietrec")
def mealRec():           
    return render_template("mealRec.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # 登录操作
        session.pop('user_id', None)
        username = request.form.get("username", None)
        password = request.form.get("password", None)
        user = [u for u in users if u.username==username] #todo 替换成数据库
        if len(user) > 0:
            user = user[0]
        if user and user.password == password:
            session['user_id'] = user.id
            return redirect(url_for('workrec'))
        
    return render_template("sign.html")
