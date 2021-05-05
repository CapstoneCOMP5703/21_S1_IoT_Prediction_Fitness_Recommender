from flask import Flask
from flask import request, render_template, redirect, url_for, session, g,flash
from dataclasses import dataclass
from datetime import timedelta

app= Flask(__name__,static_url_path="/")
app.config['SECRET_KEY'] = "sdfklas5fa2k42j"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

from SportRec_v2 import Model
rf=Model()

from Recipe_Recommendation import DietRec
dietRec = DietRec()

# from Short_term_prediction import da_rnn, dataInterpreter, contextEncoder, encoder, decoder
# import torch

import pickle
import pandas as pd
calories_cal_model=pickle.load(open('model_xgb.pkl','rb'))

@dataclass
class User:
    id: int
    user_id: int
    username: str
    password: str

users = [
	User(1, 11111116,"Admin", "Hn123456"),
	User(2, 222,"Eason", "888888"),
	User(3, 333,"Tommy", "666666"),
]

@app.before_request
def before_request():
    g.user = None
    if 'user_id' in session:
        user = [u for u in users if u.id == session['user_id']][0] #todo 替换成数据库
        g.user = user

#路由主页
@app.route("/")
def homepage():           
    return render_template("homepage.html")

#路由运动推荐，目前需要登录才能使用
@app.route("/workoutrec",methods=['GET', 'POST'])
def workoutRec():       
    if not g.user:
        return redirect(url_for('login'))    
    return render_template("workoutrec.html")

@app.route("/sportrec_model",methods=['GET', 'POST'])
def sportrec_model():
    #get calories input
    calories_get=request.form.get("calories")
    if(calories_get == ""):
        flash('Please input valid calories!')
        return render_template("workoutrec.html",)
    #parse str into int
    calories=int(calories_get) 
    if(calories > 1000):
        flash('That is to much for you, try less calories!')
        return render_template("workoutrec.html",)
    if(calories < 100):
        flash('That is not enough for you, try more calories!')
        return render_template("workoutrec.html",)

    rf.load_data_from_path('./testdata.csv')
    rf.load_model_from_path('./model_run.m', './model_bike.m', './model_mountain.m')
    data=rf.predict_data(1111116, calories)
    run_time,bike_time,mbike_time=readsplitdata(data)

    return render_template("workrec_result.html",run_time=run_time,
                            bike_time=bike_time,mbike_time=mbike_time)

def readsplitdata(data):
    length=len(data)
    run_time=""
    bike_time=""
    mbike_time=""
    for i in range(0,length):
        data_list=data[i]
        df=data_list.split(':')
        if df[0] == 'run ':
            run_time=df[1]
            print("run_time"+run_time+ " ")
        elif df[0] == 'bike ':
            bike_time=df[1]
            print("bike_time"+bike_time+ " ")
        else: 
            mbike_time=df[1]
            print("mbike_time"+mbike_time+ " ")
    return run_time,bike_time,mbike_time

#路由饮食推荐
@app.route("/dietrec",methods=['GET', 'POST'])
def mealRec():          
    return render_template("dietrec.html")

@app.route("/dietrec_model",methods=['GET', 'POST'])
def dietrec_model():
    cbox=request.values.getlist("cbox")
    calories_get=request.form.get("calories")
    if(calories_get == "" or cbox==""):
        flash('Please enter valid inputs!')
        return render_template("dietrec.html",)
    calories=int(calories_get)
    s_breakfast,s_lunch,s_dinner,s_dessert,s_vegan = 0,0,0,0,0
    count,re=0,1
    for c in cbox:
        if c =='Breakfast':
            s_breakfast=1
            count=count+1
        elif c =='Lunch':
            s_lunch=1
            count=count+1
        elif c=='Dinner':
            s_dinner=1
            count=count+1
        elif c=='Dessert':
            s_dessert=1
            count=count+1
        else:
            s_vegan=1
    
    if(count == 0):
        flash('Please choose at least one meal type!')
        return render_template("dietrec.html",)
    diet_data = dietRec.recipe_rec(calories, count,
    s_breakfast, s_lunch, s_dinner, s_dessert, s_vegan, re,0,0,0,0)
    print(diet_data)

#re增加时，其他都初始化为0
#'index_number_br', 'index_number_lun', 'index_number_din', and 'index_number_des'
    return render_template("dietrec_result.html")

#     {'Name': ['addictive and healthy granola', 'oriental edamame salad'], 
#  'Calorie_num': [251, 251], 'img_urls': ['https://images.media-allrecipes.com/userphotos/125x70/1110710.jpg , https://images.media-allrecipes.com/userphotos/560x315/1110710.jpg , ', 'https://images.media-allrecipes.com/userphotos/560x315/819709.jpg , https://images.media-allrecipes.com/userphotos/125x70/819709.jpg , https://images.media-allrecipes.com/userphotos/125x70/7079477.jpg , https://images.media-allrecipes.com/userphotos/125x70/7079476.jpg , https://images.media-allrecipes.com/userphotos/125x70/3083932.jpg , https://images.media-allrecipes.com/userphotos/125x70/2209681.jpg , https://images.media-allrecipes.com/userphotos/125x70/1120488.jpg , '], 'Meal_Type': ['breakfast', 'lunch'], 'veg': ['vegetarian', 'vegetarian']}

#路由运动记录    
@app.route("/activitylog")
def activitylog(): 
    #获得假数据
    input_data=pd.read_csv("test_calories1.csv")
    user_data=input_data.iloc[:1]
    print(user_data)
    duration_seconds = int(user_data["duration"].tolist()[0])
    distance = round(float(user_data["distance"].tolist()[0]),2)
    avg_heart_rate = round(float(user_data["avg_heart_rate"].tolist()[0]),0)
    avg_speed = round(float(user_data["avg_speed"].tolist()[0]),2)
    bike_check= int(user_data["sport_bike"].tolist()[0])
    mbike_check= int(user_data["sport_mountain bike"].tolist()[0])
    run_check= int(user_data["sport_run"].tolist()[0])
    print(distance)
    duration=cal_time(duration_seconds)
    sport_type=check_sport_type(bike_check,mbike_check,run_check)
    #mike model


    #oni model
    acc_output = calories_cal_model.predict(input_data.iloc[:1])
    actual_calories = int(acc_output)
    return render_template("activitylog.html",actual_calories=actual_calories,
    sport_type=sport_type,duration=duration,avg_speed=avg_speed,avg_heart_rate=avg_heart_rate)

def check_sport_type(bike_check,mbike_check,run_check):
    if bike_check == 1:
        return "Biking"
    elif mbike_check == 1:
        return "Mountain biking"
    else:
        return "Running"

def cal_time(seconds):
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours==0:
        return "00:%02d:%02d" % (mins, secs)
    else:
        return "%02d:%02d:%02d" % (hours, mins, secs)


#路由用户登录后显示的页面
@app.route("/profile")
def profile():        
    return render_template("profile.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    if g.user:
        return redirect(url_for('profile'))

    if request.method == 'POST':
        # login
        session.pop('user_id', None)
        username = request.form.get("username", None)
        password = request.form.get("password", None)
        print(username)
        user = [u for u in users if u.username==username] #todo 替换成数据库
        if len(user) > 0:
            user = user[0]
        if user and user.password == password:
            session['user_id'] = user.id
            return redirect(url_for('workoutRec'))
        
    return render_template("sign.html")