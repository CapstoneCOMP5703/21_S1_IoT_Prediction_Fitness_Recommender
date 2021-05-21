import pymysql
from flask import Flask
from flask import request, render_template, redirect, url_for, session, g,flash
import traceback
from dataclasses import dataclass
from datetime import timedelta,datetime
from hashlib import md5
import pandas as pd
import json

from pyecharts import options as opts
from pyecharts.charts import Line
from jinja2 import Markup

app= Flask(__name__,static_url_path="/")
app.config['SECRET_KEY'] = "sdfklasads5fa2k42j"
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)
db = pymysql.connect(host="localhost",user="root",password="961214",database="Fitastic")

from SportRec_v2 import Model
rf=Model()

from Recipe_Recommendation import DietRec
dietRec = DietRec()

# from Short_term_prediction import da_rnn, dataInterpreter, contextEncoder, encoder, decoder
from short_term_prediction_updated_v5 import da_rnn, dataInterpreter, contextEncoder, encoder, decoder,dataInterpreter_predict
import torch


import pickle
import pandas as pd
calories_cal_model=pickle.load(open('model_xgb.pkl','rb'))

# HR_track_model = torch.load('./model_heartrate_01.pt', map_location=torch.device('cpu'))
HR_track_model = torch.load('./model_epoch_04.pt')

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
    if(calories > 1000):
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
        data=rf.predict_data(session.get('userId'), calories)
    else:
        #give new user a general result
        data=rf.predict_data(1, calories)
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
    #get user input
    cbox=request.values.getlist("cbox")
    print(cbox)
    calories_get=request.form.get("calories")
    #limit user input
    if(calories_get == "" or cbox==""):
        flash('Please enter valid inputs!')
        return render_template("dietrec.html",)
    calories=int(calories_get)
    #set default value
    s_breakfast,s_lunch,s_dinner,s_dessert,s_vegan = 0,0,0,0,0
    count,re,re_breakfast,re_lunch,re_dinner,re_dessert=0,0,0,0,0,0
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
    #load DietRec model
    diet_data = dietRec.recipe_rec(calories, count,s_breakfast, s_lunch,
    s_dinner, s_dessert, s_vegan, re,re_breakfast,re_lunch,re_dinner,re_dessert)

    #save DietRec parameter into global variate
    diet_list=calories, count,s_breakfast, s_lunch,s_dinner,s_dessert, s_vegan, re,re_breakfast,re_lunch,re_dinner,re_dessert
    session['diet_list']=diet_list

    df_html_b,label_tbreakfast,df_html_l,label_tlunch,df_html_dinner,label_tdinner,df_html_dessert,label_tdessert= generateMealTable(diet_data)
    
    return render_template("dietrec_result.html",table_b_html=df_html_b,
    table_l_html=df_html_l,table_dinner_html=df_html_dinner,table_dessert_html=df_html_dessert,
    label_tbreakfast=label_tbreakfast,label_tlunch=label_tlunch,label_tdinner=label_tdinner,
    label_tdessert=label_tdessert)

def generateMealTable(diet_data):
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
        df_breakfast=generateMealDataFrame(diet_data,0,breakfast_end).T
        df_breakfast.insert(0,'images',push_img_urls(getMealImageUrls(diet_data,0,breakfast_end)))
        # use pandas method to auto generate html
        df_html_b = df_breakfast.to_html(classes="table_rec",border=0,bold_rows = bool,formatters=dict(images=path_to_image_html),header=False,index=False,escape=False) 
        label_tbreakfast = 'Recommended Breakfast'
    else:
        df_html_b=''
        label_tbreakfast=''

    if lunch_num !=0:
        flash('Lunch')
        df_lunch=generateMealDataFrame(diet_data,breakfast_end,lunch_end).T
        df_lunch.insert(0,'images',push_img_urls(getMealImageUrls(diet_data,breakfast_end,lunch_end)))
        df_html_l = df_lunch.to_html(classes="table_rec",border=0,formatters=dict(image=path_to_image_html),header=False,index=False,escape=False) 
        label_tlunch='Recommended Lunch'
    else:
        df_html_l=''
        label_tlunch=''

    if dinner_num !=0:
        flash('Dinner')
        df_dinner=generateMealDataFrame(diet_data,lunch_end,dinner_end).T
        df_dinner.insert(0,'images',push_img_urls(getMealImageUrls(diet_data, lunch_end,dinner_end)))
        df_html_dinner = df_dinner.to_html(classes="table_rec",border=0,formatters=dict(image=path_to_image_html),header=False,index=False,escape=False) 
        label_tdinner='Recommended Dinner'
    else:
        df_html_dinner=''
        label_tdinner=''
        
    if dessert_num !=0:   
        flash('Dessert')
        df_dessert=generateMealDataFrame(diet_data,dinner_end,dessert_end).T
        df_dessert.insert(0,'image',push_img_urls(getMealImageUrls(diet_data,dinner_end,dessert_end)))
        df_html_dessert = df_dessert.to_html(classes="table_rec",border=0,formatters=dict(image=path_to_image_html),header=False,index=False,escape=False) 
        label_tdessert='Recommended Dessert'
    else:
        df_html_dessert=''
        label_tdessert=''

    return df_html_b,label_tbreakfast,df_html_l,label_tlunch,df_html_dinner,label_tdinner,df_html_dessert,label_tdessert

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

def append_list(header,input_list,start,end):
    for i in range(start,end):
        header.append(input_list[i])
    return header

def append_cal(header,input_list,start,end):
    for i in range(start,end):
        data = str(input_list[i])
        data_cal = data + ' kcal'
        header.append(data_cal)
    return header

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

def generateMealDataFrame(diet_data,start,end):
    data_name=[]
    append_list(data_name,diet_data.get('Name'),start,end)

    data_Ingredients_list=[]
    append_list(data_Ingredients_list,diet_data.get('Ingredients_list'),start,end)

    data_cal=[]
    append_cal(data_cal,diet_data.get('Calorie_num'),start,end)

    df = pd.DataFrame(data=[data_name,data_cal,data_Ingredients_list])
   
    return df

@app.route("/regenerate_all",methods=['GET', 'POST'])
def regenerate_all():
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

    #load DietRec model
    diet_data = dietRec.recipe_rec(calories, count,s_breakfast, s_lunch,
    s_dinner, s_dessert, s_vegan, re,re_breakfast,re_lunch,re_dinner,re_dessert)

    #save DietRec parameter into global variate
    diet_list=calories, count,s_breakfast, s_lunch,s_dinner, s_dessert, s_vegan, re,re_breakfast,re_lunch,re_dinner,re_dessert
    session['diet_list']=diet_list

    df_html_b,label_tbreakfast,df_html_l,label_tlunch,df_html_dinner,label_tdinner,df_html_dessert,label_tdessert= generateMealTable(diet_data)
    
    return render_template("dietrec_result.html",table_b_html=df_html_b,
    table_l_html=df_html_l,table_dinner_html=df_html_dinner,table_dessert_html=df_html_dessert,
    label_tbreakfast=label_tbreakfast,label_tlunch=label_tlunch,label_tdinner=label_tdinner,
    label_tdessert=label_tdessert)
    
@app.route("/regenerate_breakfast",methods=['GET', 'POST'])
def regenerate_breakfast():
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

    #load DietRec model
    diet_data = dietRec.recipe_rec(calories, count,s_breakfast, s_lunch,
    s_dinner, s_dessert, s_vegan, re,re_breakfast,re_lunch,re_dinner,re_dessert)

    #save DietRec parameter into global variate
    diet_list=calories, count,s_breakfast, s_lunch,s_dinner, s_dessert, s_vegan, re,re_breakfast,re_lunch,re_dinner,re_dessert
    session['diet_list']=diet_list

    df_html_b,label_tbreakfast,df_html_l,label_tlunch,df_html_dinner,label_tdinner,df_html_dessert,label_tdessert= generateMealTable(diet_data)
    
    return render_template("dietrec_result.html",table_b_html=df_html_b,
    table_l_html=df_html_l,table_dinner_html=df_html_dinner,table_dessert_html=df_html_dessert,
    label_tbreakfast=label_tbreakfast,label_tlunch=label_tlunch,label_tdinner=label_tdinner,
    label_tdessert=label_tdessert)

@app.route("/regenerate_lunch",methods=['GET', 'POST'])
def regenerate_lunch():
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

    #load DietRec model
    diet_data = dietRec.recipe_rec(calories, count,s_breakfast, s_lunch,
    s_dinner, s_dessert, s_vegan, re,re_breakfast,re_lunch,re_dinner,re_dessert)

    #save DietRec parameter into global variate
    diet_list=calories, count,s_breakfast, s_lunch,s_dinner, s_dessert, s_vegan, re,re_breakfast,re_lunch,re_dinner,re_dessert
    session['diet_list']=diet_list

    df_html_b,label_tbreakfast,df_html_l,label_tlunch,df_html_dinner,label_tdinner,df_html_dessert,label_tdessert= generateMealTable(diet_data)
    
    return render_template("dietrec_result.html",table_b_html=df_html_b,
    table_l_html=df_html_l,table_dinner_html=df_html_dinner,table_dessert_html=df_html_dessert,
    label_tbreakfast=label_tbreakfast,label_tlunch=label_tlunch,label_tdinner=label_tdinner,
    label_tdessert=label_tdessert)

@app.route("/regenerate_dinner",methods=['GET', 'POST'])
def regenerate_dinner():
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

    #load DietRec model
    diet_data = dietRec.recipe_rec(calories, count,s_breakfast, s_lunch,
    s_dinner, s_dessert, s_vegan, re,re_breakfast,re_lunch,re_dinner,re_dessert)

    #save DietRec parameter into global variate
    diet_list=calories, count,s_breakfast, s_lunch,s_dinner, s_dessert, s_vegan, re,re_breakfast,re_lunch,re_dinner,re_dessert
    session['diet_list']=diet_list

    df_html_b,label_tbreakfast,df_html_l,label_tlunch,df_html_dinner,label_tdinner,df_html_dessert,label_tdessert= generateMealTable(diet_data)

    return render_template("dietrec_result.html",table_b_html=df_html_b,
    table_l_html=df_html_l,table_dinner_html=df_html_dinner,table_dessert_html=df_html_dessert,
    label_tbreakfast=label_tbreakfast,label_tlunch=label_tlunch,label_tdinner=label_tdinner,
    label_tdessert=label_tdessert)

@app.route("/regenerate_dessert",methods=['GET', 'POST'])
def regenerate_dessert():
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

    #load DietRec model
    diet_data = dietRec.recipe_rec(calories, count,s_breakfast, s_lunch,
    s_dinner, s_dessert, s_vegan, re,re_breakfast,re_lunch,re_dinner,re_dessert)

    #save DietRec parameter into global variate
    diet_list=calories, count,s_breakfast, s_lunch,s_dinner, s_dessert, s_vegan, re,re_breakfast,re_lunch,re_dinner,re_dessert
    session['diet_list']=diet_list

    meal_type=diet_data.get('Meal_Type')
    breakfast_num,lunch_num,dinner_num,dessert_num=splitMeal(meal_type)
    breakfast_end=breakfast_num
    lunch_end=breakfast_num+lunch_num
    dinner_end=breakfast_num+lunch_num+dinner_num
    dessert_end=breakfast_num+lunch_num+dinner_num+dessert_num

    #check which meal should be recommonded
    if breakfast_num != 0:
        flash('Breakfast')
        df_breakfast=generateMealDataFrame(diet_data,0,breakfast_end).T
        df_breakfast.insert(0,'images',push_img_urls(getMealImageUrls(diet_data,0,breakfast_end)))
        # use pandas method to auto generate html
        df_html_b = df_breakfast.to_html(classes="table_rec",border=0,formatters=dict(images=path_to_image_html),header=False,index=False,escape=False) 
        label_tbreakfast = 'Recommended Breakfast'
    else:
        df_html_b=''
        label_tbreakfast=''

    if lunch_num !=0:
        flash('Lunch')
        df_lunch=generateMealDataFrame(diet_data,breakfast_end,lunch_end).T
        df_lunch.insert(0,'images',push_img_urls(getMealImageUrls(diet_data,breakfast_end,lunch_end)))
        df_html_l = df_lunch.to_html(classes="table_rec",border=0,formatters=dict(image=path_to_image_html),header=False,index=False,escape=False) 
        label_tlunch='Recommended Lunch'
    else:
        df_html_l=''
        label_tlunch=''

    if dinner_num !=0:
        flash('Dinner')
        df_dinner=generateMealDataFrame(diet_data,lunch_end,dinner_end).T
        df_dinner.insert(0,'images',push_img_urls(getMealImageUrls(diet_data, lunch_end,dinner_end)))
        df_html_dinner = df_dinner.to_html(classes="table_rec",border=0,formatters=dict(image=path_to_image_html),header=False,index=False,escape=False) 
        label_tdinner='Recommended Dinner'
    else:
        df_html_dinner=''
        label_tdinner=''
        
    if dessert_num !=0:   
        flash('Dessert')
        df_dessert=generateMealDataFrame(diet_data,dinner_end,dessert_end).T
        df_dessert.insert(0,'image',push_img_urls(getMealImageUrls(diet_data,dinner_end,dessert_end)))
        df_html_dessert = df_dessert.to_html(classes="table_rec",border=0,formatters=dict(image=path_to_image_html),header=False,index=False,escape=False) 
        label_tdessert='Recommended Dessert'
    else:
        df_html_dessert=''
        label_tdessert=''
    return render_template("dietrec_result.html",table_b_html=df_html_b,
    table_l_html=df_html_l,table_dinner_html=df_html_dinner,table_dessert_html=df_html_dessert,
    label_tbreakfast=label_tbreakfast,label_tlunch=label_tlunch,label_tdinner=label_tdinner,
    label_tdessert=label_tdessert)

#路由运动记录    
@app.route("/activitylog_run")
def activitylog_run(): 
    #only logged user can use this function
    if session.get('user'):
        expected_calories = session.get('user_input_calories')
        #user should select one sport before this page
        if expected_calories == None:
            flash('Sorry, please get one recommended sport first!')
            return redirect(url_for('workoutRec'))
        else:
            flash('run')
            #get mock data
            input_data=pd.read_csv("test_calories1.csv")
            user_data=input_data.iloc[:10]
            duration_seconds = int(user_data["duration"].tolist()[3])
            distance = round(float(user_data["distance"].tolist()[3]),2)
            avg_heart_rate = round(float(user_data["avg_heart_rate"].tolist()[3]),0)
            avg_speed = round(float(user_data["avg_speed"].tolist()[3]),2)
            bike_check= int(user_data["sport_bike"].tolist()[3])
            mbike_check= int(user_data["sport_mountain bike"].tolist()[3])
            run_check= int(user_data["sport_run"].tolist()[3])
            duration=cal_time(duration_seconds)
            sport_type=check_sport_type(bike_check,mbike_check,run_check)
            
            system_time=str(datetime.now())
            time=system_time.split('.')[0]

            #HR-track model
            use_cuda=torch.cuda.is_available() 
            HR_output = HR_track_model.predict(id = 605119015)
            hr_min=int(min(min(HR_output[0][0]),min(HR_output[0][1]))-10)
            hr_max=int(max(max(HR_output[0][0]),max(HR_output[0][1]))+10)

            #get mock speed data
            speed_altitude_mock = pd.read_csv("speed_altitude_mock.csv")

            #1 for bike, 2 for mbike, 3 for run
            mock_data = speed_altitude_mock.iloc[2]
            sport = mock_data["sport"]
            speed = mock_data["speed"]
            altitude = mock_data["altitude"]
            #set echarts
            x = []
            for i in range(50):
                x.append('')
            #Fit-track model
            acc_output = calories_cal_model.predict(input_data.iloc[3:4])
            actual_calories = int(acc_output)
            return render_template("activitylog.html",actual_calories=actual_calories,
            time=time,distance=distance,sport_type=sport_type,duration=duration,avg_speed=avg_speed,
            avg_heart_rate=avg_heart_rate,expected_calories=expected_calories,
            heartrate_pre=json.dumps(HR_output[0][0]),heartrate_tar=json.dumps(HR_output[0][1]),
            xaxis=Markup(json.dumps(x)),altitude=Markup(altitude),speed=Markup(speed),
            hr_max=hr_max,hr_min=hr_min,
            sport=Markup(json.dumps(sport))
            )   
    else:
        flash('Sorry, please log in to use this function!')
        return redirect(url_for('login'))

@app.route("/activitylog_bike")
def activitylog_bike(): 
    #only logged user can use this function
    if session.get('user'):
        expected_calories = session.get('user_input_calories')
        #user should select one sport before this page
        if expected_calories == None:
            flash('Sorry, please get one recommended sport first!')
            return redirect(url_for('workoutRec'))
        else:
            flash('bike')
            #get mock data
            input_data=pd.read_csv("test_calories1.csv")
            user_data=input_data.iloc[:10]
            duration_seconds = int(user_data["duration"].tolist()[7])
            distance = round(float(user_data["distance"].tolist()[7]),2)
            avg_heart_rate = round(float(user_data["avg_heart_rate"].tolist()[7]),0)
            avg_speed = round(float(user_data["avg_speed"].tolist()[7]),2)
            bike_check= int(user_data["sport_bike"].tolist()[7])
            mbike_check= int(user_data["sport_mountain bike"].tolist()[7])
            run_check= int(user_data["sport_run"].tolist()[7])
            duration=cal_time(duration_seconds)
            sport_type=check_sport_type(bike_check,mbike_check,run_check)
            
            system_time=str(datetime.now())
            time=system_time.split('.')[0]

            #HR-track model
            use_cuda=torch.cuda.is_available() 
            HR_output = HR_track_model.predict(id = 328420333)
            hr_min=int(min(min(HR_output[0][0]),min(HR_output[0][1]))-10)
            hr_max=int(max(max(HR_output[0][0]),max(HR_output[0][1]))+10)

            #get mock speed data
            speed_altitude_mock = pd.read_csv("speed_altitude_mock.csv")

            #0 for bike, 1 for mbike, 2 for run
            mock_data = speed_altitude_mock.iloc[0]
            sport = mock_data["sport"]
            speed = mock_data["speed"]
            altitude = mock_data["altitude"]
            #set echarts
            x = []
            for i in range(50):
                x.append('')
            #Fit-track model
            acc_output = calories_cal_model.predict(input_data.iloc[7:8])
            actual_calories = int(acc_output)
            return render_template("activitylog.html",actual_calories=actual_calories,
            time=time,distance=distance,sport_type=sport_type,duration=duration,avg_speed=avg_speed,
            avg_heart_rate=avg_heart_rate,expected_calories=expected_calories,
            heartrate_pre=json.dumps(HR_output[0][0]),heartrate_tar=json.dumps(HR_output[0][1]),
            xaxis=Markup(json.dumps(x)),altitude=Markup(altitude),speed=Markup(speed),
            hr_max=hr_max,hr_min=hr_min,
            sport=Markup(json.dumps(sport))
            )   
    else:
        flash('Sorry, please log in to use this function!')
        return redirect(url_for('login'))

@app.route("/activitylog_mbike")
def activitylog_mbike(): 
    #only logged user can use this function
    if session.get('user'):
        expected_calories = session.get('user_input_calories')
        #user should select one sport before this page
        if expected_calories == None:
            flash('Sorry, please get one recommended sport first!')
            return redirect(url_for('workoutRec'))
        else:
            flash('mbike')
            #get mock data
            input_data=pd.read_csv("test_calories1.csv")
            user_data=input_data.iloc[:10]
            duration_seconds = int(user_data["duration"].tolist()[5])
            distance = round(float(user_data["distance"].tolist()[5]),2)
            avg_heart_rate = round(float(user_data["avg_heart_rate"].tolist()[5]),0)
            avg_speed = round(float(user_data["avg_speed"].tolist()[5]),2)
            bike_check= int(user_data["sport_bike"].tolist()[5])
            mbike_check= int(user_data["sport_mountain bike"].tolist()[5])
            run_check= int(user_data["sport_run"].tolist()[5])
            duration=cal_time(duration_seconds)
            sport_type=check_sport_type(bike_check,mbike_check,run_check)
            
            system_time=str(datetime.now())
            time=system_time.split('.')[0]

            #HR-track model
            use_cuda=torch.cuda.is_available() 
            HR_output = HR_track_model.predict(id = 373904132)
            hr_min=int(min(min(HR_output[0][0]),min(HR_output[0][1]))-10)
            hr_max=int(max(max(HR_output[0][0]),max(HR_output[0][1]))+10)

            #get mock speed data
            speed_altitude_mock = pd.read_csv("speed_altitude_mock.csv")

            #1 for bike, 2 for mbike, 3 for run
            mock_data = speed_altitude_mock.iloc[1]
            sport = mock_data["sport"]
            speed = mock_data["speed"]
            altitude = mock_data["altitude"]
            #set echarts
            x = []
            for i in range(50):
                x.append('')
            #Fit-track model
            acc_output = calories_cal_model.predict(input_data.iloc[5:6])
            actual_calories = int(acc_output)
            return render_template("activitylog.html",actual_calories=actual_calories,
            time=time,distance=distance,sport_type=sport_type,duration=duration,avg_speed=avg_speed,
            avg_heart_rate=avg_heart_rate,expected_calories=expected_calories,
            heartrate_pre=json.dumps(HR_output[0][0]),heartrate_tar=json.dumps(HR_output[0][1]),
            xaxis=Markup(json.dumps(x)),altitude=Markup(altitude),speed=Markup(speed),
            hr_max=hr_max,hr_min=hr_min,
            sport=Markup(json.dumps(sport))
            )   
    else:
        flash('Sorry, please log in to use this function!')
        return redirect(url_for('login'))

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

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        cursor = db.cursor()
        username = request.form.get('username', None)
        password = request.form.get('password', None)
        sql = "select * from users where username= '{}'".format(username, encoding='utf-8') 
        try:
            # 执行sql语句
            cursor.execute(sql)
            result = cursor.fetchall()
            if (len(result)==0):
                flash("The username does not exist!")
                return redirect(url_for('login'))          
            else:
                if result[0][2] == md5(password.encode('utf-8')).hexdigest():
                    print("Login successfully!") 
                    session['userId'] = result[0][0]
                    session['user'] = request.form.get('username', None)
                    return redirect(url_for('workoutRec'))
                else:
                    flash("Username or password is wrong!")
                    return redirect(url_for('login'))
            # 提交到数据库执行
            db.commit()
        except:
            # 如果发生错误则回滚
            traceback.print_exc()
            db.rollback()
        # 关闭数据库连接
        db.close()
    return render_template("sign.html")

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        cursor = db.cursor()
        username = request.form.get('username', None)
        email = request.form.get('email', None)
        password = request.form.get('password', None)
        repassword = request.form.get('repassword', None)
        sql = "select count(*) from users where username= '{}'".format(username, encoding='utf-8')
        try:
            cursor.execute(sql)
            result = cursor.fetchone()
            if result[0]== 0:
                if password != repassword:
                    flash("Inconsistency of password!")
                    return redirect(url_for('signup'))
                sql = 'INSERT INTO users (username, password, email) VALUES ("{}", "{}", "{}")'.format(username,md5(password.encode('utf-8')).hexdigest(), email, encoding='utf-8')
                cursor.execute(sql)
                db.commit()
                flash("Register successfully, please sign in")
                return redirect(url_for('login'))
            else:
                if result[0]!= 0:
                    flash("please sign up with another name") 
                    return redirect(url_for('signup'))
                db.commit()
        except Exception as err:
            print(err)
        # 关闭数据库连接    
        db.close()
    return render_template("signup.html")

@app.route("/reset", methods=['GET', 'POST'])
def reset():
    if request.method == 'POST':     
        cursor = db.cursor()
        username = request.form.get('username', None)
        email = request.form.get('email', None)
        password = request.form.get('password', None)
        repassword = request.form.get('repassword', None)
        if len(password)>=6:
            if password == repassword:
                sql = "Update users SET password = '{}' WHERE username = '{}'".format(md5(password.encode('utf-8')).hexdigest(), username, encoding='utf-8') 
                sql_2 = "Select * from users where username= '{}'".format(username, encoding='utf-8') 
                try:          
                    # 执行sql语句
                    cursor.execute(sql_2)
                    result = cursor.fetchall()
                    if (len(result)==0):
                        flash("The username does not exist!") 
                        return redirect(url_for('reset'))     
                    else:
                        if result[0][3] == email:
                            cursor.execute(sql)
                            db.commit()
                            print("Update successfully!") 
                            return redirect(url_for('login'))
                        else:
                            flash("Wrong Email!")
                            return redirect(url_for('reset'))         
                except:
                    # 如果发生错误则回滚
                    traceback.print_exc()
                    db.rollback()
                # 关闭数据库连接
                db.close()
            else:
                flash('Inconsistency of password!')
                return redirect(url_for('reset'))
        else:
            flash("Password should be at least 6 characters in length")
            return redirect(url_for('reset'))
    return render_template("reset.html")  

@app.route("/logout")
def logout():
    session.clear()
    return render_template("homepage.html") 

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
    