import flask
import pymysql
from pymysql.cursors import DictCursor
from flask import Flask
from flask import request, render_template, redirect, url_for, session,g, flash,jsonify
import traceback
from dataclasses import dataclass
from datetime import timedelta,datetime
import hashlib
import pandas as pd
import json
import numpy as np
from decimal import *

from pyecharts import options as opts
from pyecharts.charts import Line
from jinja2 import Markup

from config import config

app= Flask(__name__,static_url_path="/")
app.config['SECRET_KEY'] = "sdfklasads5fa2k42j"
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)
# db = pymysql.connect(host=config.host,port=config.port, user=config.user, password=config.password, database=config.database)
db = pymysql.connect(host="sh-cdb-rle6a9ic.sql.tencentcdb.com",port=59992,user="root",password="capstone25_2",database="Fitastic")

from SportRec_v2 import Model
rf=Model()

from Recipe_Recommendation import DietRec
from DetailsDisplay import DetailsDisplay
dietRec = DietRec()
detailsDisplay = DetailsDisplay()

# from Short_term_prediction import da_rnn, dataInterpreter, contextEncoder, encoder, decoder
from short_term_prediction_updated_v5 import da_rnn, dataInterpreter, contextEncoder, encoder, decoder,dataInterpreter_predict
import torch

import pickle
import pandas as pd
calories_cal_model=pickle.load(open('model_xgb.pkl','rb'))

# HR_track_model = torch.load('./model_heartrate_01.pt', map_location=torch.device('cpu'))
HR_track_model = torch.load('./model_epoch_04.pt')

userId=7178673  

#homepage
@app.route("/")
def homepage():      
    return render_template("homepage.html")

#SportRec page
@app.route("/workoutrec",methods=['GET', 'POST'])
def workoutRec():       
    return render_template("workoutrec.html")

#SportRec result page
@app.route("/sportrec_model",methods=['GET', 'POST'])
def sportrec_model():
    #get calories input
    calories_get=request.form.get("calories")
    #limit user input
    if(calories_get == ""):
        flash('Please input valid calories!')
        return render_template("workoutrec.html",)
    #parse str into int
    calories=int(calories_get) 
    #limit user input
    if(calories > 2001):
        flash('That is too much for you, try fewer calories!')
        return render_template("workoutrec.html",)
    if(calories < 100):
        flash('That is not enough for you, try more calories!')
        return render_template("workoutrec.html",)
    #save user input into session
    session['user_input_calories']=calories
    #load SportRec model
    rf.load_data_from_path('./testdata.csv')
    rf.load_model_from_path('./model_run.m', './model_bike.m', './model_mountain.m')
    if session.get('user'):
        #get the userId to predict personalized result
        data=rf.predict_data(userId, calories)
        # data=rf.predict_data(session.get('userId'), calories)
    else:
        #give new user a general result
        data=rf.predict_data(1, calories)
        # data=rf.predict_data(1, calories)
        flash('//todo 解释这是新用户的托底数据')
    
    #get the sport duration
    run_time,bike_time,mbike_time=readsplitdata(data)
    #give front-end indicators to show which recommondation
    if run_time != "":
        flash('run')
    if bike_time != "":
        flash('bike')
    if mbike_time != "":
        flash('mbike')

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
        elif df[0] == 'bike ':
            bike_time=df[1]
        else: 
            mbike_time=df[1]
    return run_time,bike_time,mbike_time

#DietRec page
@app.route("/dietrec",methods=['GET', 'POST'])
def mealRec():          
    return render_template("dietrec.html")

#DietRec result page
@app.route("/dietrec_model",methods=['GET', 'POST'])
def dietrec_model():
    #set default value
    s_breakfast,s_lunch,s_dinner,s_dessert,s_vegan = 0,0,0,0,0
    count,re,re_breakfast,re_lunch,re_dinner,re_dessert=0,0,0,0,0,0

    if request.form.get("hidden") == "regenerate_all":
        print("regenerate_all")
        diet_list=session.get('diet_list')
        calories = diet_list[0]
        count= diet_list[1]
        s_breakfast =diet_list[2]
        s_lunch=diet_list[3]
        s_dinner =diet_list[4]
        s_dessert=diet_list[5]
        s_vegan=diet_list[6]
        re=diet_list[7]
        #when user regenerate all meals, set other re into 0
        re_breakfast=0
        re_lunch=0
        re_dinner=0
        re_dessert=0
        re=re+1
        diet_list=calories, count,s_breakfast, s_lunch,s_dinner,s_dessert, s_vegan, re,re_breakfast,re_lunch,re_dinner,re_dessert
        session['diet_list']=diet_list
    elif request.form.get("hidden") == "regenerate_breakfast":
        print("regenerate_breakfast")
        diet_list=session.get('diet_list')
        calories = diet_list[0]
        count= diet_list[1]
        s_breakfast =diet_list[2]
        s_lunch=diet_list[3]
        s_dinner =diet_list[4]
        s_dessert=diet_list[5]
        s_vegan=diet_list[6]
        re=diet_list[7]
        re_breakfast=diet_list[8]
        re_lunch=diet_list[9]
        re_dinner=diet_list[10]
        re_dessert=diet_list[11]
        re_breakfast=re_breakfast+1
        diet_list=calories, count,s_breakfast, s_lunch,s_dinner,s_dessert, s_vegan, re,re_breakfast,re_lunch,re_dinner,re_dessert
        session['diet_list']=diet_list
    elif request.form.get("hidden") == "regenerate_lunch":
        print("regenerate_lunch")
        diet_list=session.get('diet_list')
        calories = diet_list[0]
        count= diet_list[1]
        s_breakfast =diet_list[2]
        s_lunch=diet_list[3]
        s_dinner =diet_list[4]
        s_dessert=diet_list[5]
        s_vegan=diet_list[6]
        re=diet_list[7]
        re_breakfast=diet_list[8]
        re_lunch=diet_list[9]
        re_dinner=diet_list[10]
        re_dessert=diet_list[11]
        re_lunch=re_lunch+1
        diet_list=calories, count,s_breakfast, s_lunch,s_dinner,s_dessert, s_vegan, re,re_breakfast,re_lunch,re_dinner,re_dessert
        session['diet_list']=diet_list
    elif request.form.get("hidden") == "regenerate_dinner":
        print("regenerate_dinner")
        diet_list=session.get('diet_list')
        calories = diet_list[0]
        count= diet_list[1]
        s_breakfast =diet_list[2]
        s_lunch=diet_list[3]
        s_dinner =diet_list[4]
        s_dessert=diet_list[5]
        s_vegan=diet_list[6]
        re=diet_list[7]
        re_breakfast=diet_list[8]
        re_lunch=diet_list[9]
        re_dinner=diet_list[10]
        re_dessert=diet_list[11]
        re_dinner=re_dinner+1
        diet_list=calories, count,s_breakfast, s_lunch,s_dinner,s_dessert, s_vegan, re,re_breakfast,re_lunch,re_dinner,re_dessert
        session['diet_list']=diet_list
    elif request.form.get("hidden") == "regenerate_dessert":
        print("regenerate_dessert")
        diet_list=session.get('diet_list')
        calories = diet_list[0]
        count= diet_list[1]
        s_breakfast =diet_list[2]
        s_lunch=diet_list[3]
        s_dinner =diet_list[4]
        s_dessert=diet_list[5]
        s_vegan=diet_list[6]
        re=diet_list[7]
        re_breakfast=diet_list[8]
        re_lunch=diet_list[9]
        re_dinner=diet_list[10]
        re_dessert=diet_list[11]
        re_dessert=re_dessert+1
        diet_list=calories, count,s_breakfast, s_lunch,s_dinner,s_dessert, s_vegan, re,re_breakfast,re_lunch,re_dinner,re_dessert
        session['diet_list']=diet_list
    elif request.form.get("hidden") == "close":
        diet_list=session.get('diet_list')
        calories = diet_list[0]
        count= diet_list[1]
        s_breakfast =diet_list[2]
        s_lunch=diet_list[3]
        s_dinner =diet_list[4]
        s_dessert=diet_list[5]
        s_vegan=diet_list[6]
        re=diet_list[7]
        re_breakfast=diet_list[8]
        re_lunch=diet_list[9]
        re_dinner=diet_list[10]
        re_dessert=diet_list[11]
        diet_list=calories, count,s_breakfast, s_lunch,s_dinner,s_dessert, s_vegan, re,re_breakfast,re_lunch,re_dinner,re_dessert
        session['diet_list']=diet_list
    else:     
        #get user input
        cbox=request.values.getlist("cbox")
        calories_get=request.form.get("calories")
        #limit user input
        if(calories_get == "" or cbox==""):
            flash('Please enter valid inputs!')
            return render_template("dietrec.html",)
        calories=int(calories_get)
        #read user input
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
        #limit user input
        if(count == 0):
            flash('Please choose at least one meal type!')
            return render_template("dietrec.html",)

        diet_list=calories, count,s_breakfast, s_lunch,s_dinner,s_dessert, s_vegan, re,re_breakfast,re_lunch,re_dinner,re_dessert
        session['diet_list']=diet_list

    #load DietRec model
    diet_data = dietRec.recipe_rec(calories, count,s_breakfast, s_lunch,
    s_dinner, s_dessert, s_vegan, re,re_breakfast,re_lunch,re_dinner,re_dessert)

    #save DietRec parameter into global variate

    df_html_b,df_html_l,df_html_dinner,df_html_dessert,sum_breakfast_cal,sum_lunch_cal,sum_dinner_cal,sum_dessert_cal= generateMealTableNoIgre(diet_data)
   
    total_calories= int(sum_breakfast_cal)+int(sum_lunch_cal)+int(sum_dinner_cal)+int(sum_dessert_cal)

    return render_template("dietrec_result.html",table_b_html=df_html_b,
    table_l_html=df_html_l,table_dinner_html=df_html_dinner,table_dessert_html=df_html_dessert,
    sum_breakfast_cal=sum_breakfast_cal,sum_lunch_cal=sum_lunch_cal,
    sum_dinner_cal=sum_dinner_cal,sum_dessert_cal=sum_dessert_cal,total_calories=total_calories
    )

@app.route("/food_details",methods=['GET', 'POST'])
def food_details(): 
    food_name=request.form.get("hidden")
    ingredient_list,direction_list,img_url,calorie,prep_time,cook_time,meal_type,index_value  = detailsDisplay.details_display(food_name)
    ingredient_list=ingredient_list[index_value[0]]
    direction_list=direction_list[index_value[0]]
    # img_url[index_value[0]]
    # calorie[index_value[0]]
    prep_time=prep_time[index_value[0]]
    cook_time=cook_time[index_value[0]]
    # meal_type[index_value[0]])
    return render_template("food_details.html",food_name=food_name,ingredient_list=ingredient_list,
    direction_list=direction_list,prep_time=prep_time,cook_time=cook_time
    )

def generateMealTableNoIgre(diet_data):
    meal_type=diet_data.get('Meal_Type')
    #read how many meals user selected
    breakfast_num,lunch_num,dinner_num,dessert_num=splitMeal(meal_type)
    breakfast_end=breakfast_num
    lunch_end=breakfast_num+lunch_num
    dinner_end=breakfast_num+lunch_num+dinner_num
    dessert_end=breakfast_num+lunch_num+dinner_num+dessert_num

    #check which meal should be recommonded
    if breakfast_num != 0:
        flash('Breakfast')
        df_breakfast,sum_breakfast_cal=generateMealDataFrameNoIgre(diet_data,0,breakfast_end)
        df_breakfast=df_breakfast.T
        df_breakfast.insert(0,'images',push_img_urls(getMealImageUrls(diet_data,0,breakfast_end)))
        df_breakfast.insert(3,'button',generateMealNameNoIgre(diet_data,0,breakfast_end))
        # use pandas method to auto generate html
        df_html_b = df_breakfast.to_html(classes="table_rec",border=0,bold_rows = bool,formatters=dict(images=path_to_image_html,button=value_to_button_html),header=False,index=False,escape=False) 
    else:
        sum_breakfast_cal=0
        df_html_b=''

    if lunch_num !=0:
        flash('Lunch')
        df_lunch,sum_lunch_cal=generateMealDataFrameNoIgre(diet_data,breakfast_end,lunch_end)
        df_lunch=df_lunch.T
        df_lunch.insert(0,'images',push_img_urls(getMealImageUrls(diet_data,breakfast_end,lunch_end)))
        df_lunch.insert(3,'button',generateMealNameNoIgre(diet_data,breakfast_end,lunch_end))
        df_html_l = df_lunch.to_html(classes="table_rec",border=0,formatters=dict(image=path_to_image_html,button=value_to_button_html),header=False,index=False,escape=False) 
    else:
        sum_lunch_cal=0
        df_html_l=''

    if dinner_num !=0:
        flash('Dinner')
        df_dinner,sum_dinner_cal=generateMealDataFrameNoIgre(diet_data,lunch_end,dinner_end)
        df_dinner=df_dinner.T
        df_dinner.insert(0,'images',push_img_urls(getMealImageUrls(diet_data, lunch_end,dinner_end)))
        df_dinner.insert(3,'button',generateMealNameNoIgre(diet_data,lunch_end,dinner_end))
        df_html_dinner = df_dinner.to_html(classes="table_rec",border=0,formatters=dict(image=path_to_image_html,button=value_to_button_html),header=False,index=False,escape=False) 
    else:
        sum_dinner_cal=0
        df_html_dinner=''
        
    if dessert_num !=0:   
        flash('Dessert')
        df_dessert,sum_dessert_cal=generateMealDataFrameNoIgre(diet_data,dinner_end,dessert_end)
        df_dessert=df_dessert.T
        df_dessert.insert(0,'image',push_img_urls(getMealImageUrls(diet_data,dinner_end,dessert_end)))
        df_dessert.insert(3,'button',generateMealNameNoIgre(diet_data,dinner_end,dessert_end))
        df_html_dessert = df_dessert.to_html(classes="table_rec",border=0,formatters=dict(image=path_to_image_html,button=value_to_button_html),header=False,index=False,escape=False) 
    else:
        sum_dessert_cal=0
        df_html_dessert=''

    return df_html_b,df_html_l,df_html_dinner,df_html_dessert,sum_breakfast_cal,sum_lunch_cal,sum_dinner_cal,sum_dessert_cal

def push_img_urls(images):
    image_url=[]
    for image in images:
        image_to_html=path_to_image_html(image)
        image_url.append(image_to_html)
    return image_url

def getMealImageUrls(diet_data,start,end):
    img_list=[]
    data_img_urls=diet_data.get('img_urls') 
    for i in range(start,end):
        img_list.append(data_img_urls[i])
    return img_list

def path_to_image_html(path):
    return '<img src="'+ path + '" width="60"   background-position: center center; background-size: cover;>'

def value_to_button_html(value):
    ingredient_list,direction_list,img_url,calorie,prep_time,cook_time,meal_type,index_value  = detailsDisplay.details_display(value)
    ingredient_list=ingredient_list[index_value[0]]
    direction_list=direction_list[index_value[0]]
    cook_time=cook_time[index_value[0]]
    prep_time=prep_time[index_value[0]]
    img_url=img_url[index_value[0]]
    calorie=str(calorie[index_value[0]])
    return "<input type='button'  id='details_button' href = 'javascript:void(0)' onclick = 'popWin(&apos;"+ value +"&apos;,&apos;"+ ingredient_list +"&apos;,&apos;"+ direction_list +"&apos;,&apos;"+ cook_time +"&apos;,&apos;"+ prep_time +"&apos;,&apos;"+ img_url +"&apos;,&apos;"+ calorie +"&apos;);' value='Details'></input>"

def append_list(header,input_list,start,end):
    for i in range(start,end):
        header.append(input_list[i])
    return header

def append_cal(header,input_list,start,end):
    sum_cal=0
    for i in range(start,end):
        data = str(input_list[i])
        data_cal = data + ' kcal'
        header.append(data_cal)
        sum_cal = sum_cal+int(data)
    return header,sum_cal

def splitMeal(data):
    length=len(data)
    breakfast_num=0
    lunch_num=0
    dinner_num=0
    dessert_num=0
    for i in range(0,length):
        data_list=data[i]
        if data_list == 'breakfast':
            breakfast_num=breakfast_num+1
        elif data_list == 'lunch':
            lunch_num=lunch_num+1
        elif data_list == 'dinner':
            dinner_num=dinner_num+1
        else: 
            dessert_num=dessert_num+1
    return breakfast_num,lunch_num,dinner_num,dessert_num

def generateMealDataFrameNoIgre(diet_data,start,end):
    data_name=[]
    append_list(data_name,diet_data.get('Name'),start,end)

    data_cal=[]
    data_cal,sum_cal=append_cal(data_cal,diet_data.get('Calorie_num'),start,end)

    df = pd.DataFrame(data=[data_name,data_cal])
   
    return df,sum_cal

def generateMealNameNoIgre(diet_data,start,end):
    data_name=[]
    data_names=diet_data.get('Name') 
    for i in range(start,end):
        data_name.append(data_names[i])
    return data_name


#路由运动记录    
@app.route("/activitylog",methods=['GET', 'POST'])
def activitylog(): 
    #only logged user can use this function
    if session.get('user'):
        expected_calories = session.get('user_input_calories')
        # userId=session.get('userId')
        #user should select one sport before this page
        if expected_calories == None:
            flash('Sorry, please get one recommended sport first!')
            return redirect(url_for('workoutRec'))
        else:
            data = pd.read_csv('mock_dataset.csv')
            
            if request.form.get("hidden")=="run":
                print("user selected run!")
                flash('run')
                sport_type='run'
                calories = data.Calories[(data.User_Id==userId) & (data.Sport_run==1)].tolist()
                print("calories",calories)
                calories_target = index_number(calories,expected_calories)
                workoutId= int(data.Id[(data.User_Id==userId) & (data.Calories==calories_target) & (data.Sport_run==1)])
                print("workoutId",workoutId)
            elif request.form.get("hidden")=="bike":
                print("user selected bike!")
                flash('bike')
                sport_type='bike'
                calories = data.Calories[(data.User_Id==userId) & (data.Sport_bike==1)].tolist()
                calories_target = index_number(calories,expected_calories)
                workoutId= int(data.Id[(data.User_Id==userId) & (data.Calories==calories_target) & (data.Sport_bike==1)])
                print("workoutId",workoutId)
            elif request.form.get("hidden")=="mbike":
                print("user selected mbike!")
                flash('mbike')
                sport_type='mbike'
                calories = data.Calories[(data.User_Id==userId) & (data.Sport_mountain_bike==1)].tolist()
                calories_target = index_number(calories,expected_calories)
                workoutId= int(data.Id[(data.User_Id==userId) & (data.Calories==calories_target) & (data.Sport_mountain_bike==1)])
                print("workoutId",workoutId)
            else:
                flash('Please retype calories!')
                return redirect(url_for('workoutRec'))
            
            speed = data.Speed_Adjusted[data.Id==workoutId].tolist()[0]
            altitude =data.Altitude[data.Id==workoutId].tolist()[0]

            duration_seconds = data.Duration[data.Id==workoutId].tolist()[0]
            duration=cal_time(duration_seconds)
            
            distance_meter = data.Distance[data.Id==workoutId].tolist()[0]
            distance=round(float(distance_meter))

            avg_heart_rate_original=data.avg_heart_rate[data.Id==workoutId].tolist()[0]
            avg_heart_rate = round(float(avg_heart_rate_original))

            avg_speed_original=data.avg_speed[data.Id==workoutId].tolist()[0]
            avg_speed=round(float(avg_speed_original))

            bike_check= int(data.Sport_bike[data.Id==workoutId].tolist()[0])
            mbike_check= int(data. Sport_mountain_bike[data.Id==workoutId].tolist()[0])
            run_check= int(data.Sport_run[data.Id==workoutId].tolist()[0])
            sport_type=check_sport_type(bike_check,mbike_check,run_check)

            user_data = pd.DataFrame([[workoutId,userId,duration_seconds,distance_meter,avg_heart_rate_original,avg_speed_original,bike_check,mbike_check,run_check]], columns=['id', 'userId','duration','distance','avg_heart_rate','avg_speed','sport_bike','sport_mountain bike','sport_run'])
            print(user_data)

            system_time=str(datetime.now())
            time=system_time.split('.')[0]

            #Fit-track model
            acc_output = calories_cal_model.predict(user_data)
            print("acc_output",acc_output)
            actual_calories = int(acc_output)
            
            #HR-track model
            use_cuda=torch.cuda.is_available() 
            HR_output = HR_track_model.predict(id = workoutId)
            hr_min=int(min(min(HR_output[0][0]),min(HR_output[0][1]))-10)
            hr_max=int(max(max(HR_output[0][0]),max(HR_output[0][1]))+10)

            #set echarts
            x = []
            for i in range(290):
                x.append('')


            return render_template("activitylog.html",actual_calories=actual_calories,
            time=time,distance=distance,sport_type=sport_type,duration=duration,avg_speed=avg_speed,
            avg_heart_rate=avg_heart_rate,expected_calories=expected_calories,
            heartrate_pre=json.dumps(HR_output[0][0]),heartrate_tar=json.dumps(HR_output[0][1]),
            xaxis=Markup(json.dumps(x)),altitude=Markup(altitude),speed=Markup(speed),
            hr_max=hr_max,hr_min=hr_min
            # sport=Markup(json.dumps(sport))
            )   
    else:
        flash('Sorry, please log in to use this function!')
        return redirect(url_for('login'))

def index_number(a,b):
    return  min(a, key=lambda x: abs(x - b))

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

# log in
@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        cursor = db.cursor(DictCursor)
        # get input values from form
        username = request.form.get('username', None)
        password = request.form.get('password', None)

        # sql statement
        sql = "select * from users where username= %s"
        try:
            # execute sql statement
            cursor.execute(sql, username)
            result = cursor.fetchall()
            # when no result found in the database
            if (len(result)==0):
                flash("The username does not exist!")
                return redirect(url_for('login'))          
            # when one result found in the database
            else:
                # encrypt the password
                if result[0]["password"] == hashlib.sha512(password.encode('utf-8')).hexdigest():
                    # store the session
                    session['userId'] = result[0]["user_id"]
                    session['user'] = request.form.get('username', None)
                    return redirect(url_for('workoutRec'))
                else:
                    flash("Username or password is wrong!")
                    return redirect(url_for('login'))
            db.commit()
        except:
            # rollback when mistake
            traceback.print_exc()
            db.rollback()
        # close the database connection
        db.close()
    return render_template("sign.html")

# sign up
@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        cursor = db.cursor(DictCursor)
        # get input values from form
        username = request.form.get('username', None)
        email = request.form.get('email', None)
        password = request.form.get('password', None)
        repassword = request.form.get('repassword', None)

        # sql statement
        sql = "select count(*) from users where username= %s"
        try:
            # execute sql statement
            cursor.execute(sql, username)
            result = cursor.fetchall()

            # no same name found in the database
            if result[0]["count(*)"]== 0:
                if password != repassword:
                    flash("Inconsistency of password!")
                    return redirect(url_for('signup'))
                else:
                    # sql statement for adding new user to the database
                    sql = 'INSERT INTO users (username, password, email) VALUES (%s, %s, %s)'
                    cursor.execute(sql,(username,hashlib.sha512(password.encode('utf-8')).hexdigest(),email))
                    db.commit()
                    flash("Register successfully, please sign in")
                    return redirect(url_for('login'))
            # same name found in the database
            else:
                flash("please sign up with another name") 
                return redirect(url_for('signup'))
                db.commit()
        except:
            # rollback when mistake
            traceback.print_exc()
            db.rollback()
        # close the database connection   
        db.close()
    return render_template("signup.html")

# password reset
@app.route("/reset", methods=['GET', 'POST'])
def reset():
    if request.method == 'POST':     
        cursor=db.cursor(DictCursor)
        # get input values from form
        username = request.form.get('username', None)
        email = request.form.get('email', None)
        password = request.form.get('password', None)
        repassword = request.form.get('repassword', None)
        
        try:   
            # when the length of password is more than 5
            if len(password)>=6:
                # when confirmed password is same as the password 
                if password == repassword:
                    # sql statement for updating the password
                    sql = "Update users SET password = %s WHERE username = %s"
                    sql_2 = "Select * from users where username= %s"       
                    # execute sql statement 
                    cursor.execute(sql_2, username)
                    result = cursor.fetchall()
                
                    # when no result found in the database
                    if (len(result)==0):
                        flash("The username does not exist!") 
                        return redirect(url_for('reset'))     
                    else:
                    # verifying email
                        if result[0]["email"] == email:
                            cursor.execute(sql,(hashlib.sha512(password.encode('utf-8')).hexdigest(), username))
                            db.commit()
                            return redirect(url_for('login'))
                        else:
                            flash("Wrong Email!")
                            return redirect(url_for('reset')) 
                else:
                    flash('Inconsistency of password!')
                    return redirect(url_for('reset'))
            else:
                flash("Password should be at least 6 characters in length")
                return redirect(url_for('reset'))                     
        except:
            # rollback when mistake
            traceback.print_exc()
            db.rollback()
        # close the database connection 
        db.close()

    return render_template("reset.html")  

# log out
@app.route("/logout")
def logout():
    # clear session
    session.clear()
    return render_template("homepage.html")
    
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=3000)


#todo
# 1.  unit testing                     （peter可能不做） https://pypi.org/project/flask-unittest/
# 2.  check image的版权 cc0 license    （oni）
# 3.  draft report                       （whole team）
# 4.  换data的instruction                （monica，oni，mike，demi）
# 5.  Usability testing
# 6.  数据库                            （demi）


