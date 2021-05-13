import pymysql
from flask import Flask
from flask import request, render_template, redirect, url_for, session, g,flash
import traceback
from dataclasses import dataclass
from datetime import timedelta,datetime
from hashlib import md5

from pyecharts import options as opts
from pyecharts.charts import Line

app= Flask(__name__,static_url_path="/")
app.config['SECRET_KEY'] = "sdfklasads5fa2k42j"
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)

from SportRec_v2 import Model
rf=Model()

from Recipe_Recommendation import DietRec
dietRec = DietRec()

from Short_term_prediction import da_rnn, dataInterpreter, contextEncoder, encoder, decoder
import torch


import pickle
import pandas as pd
calories_cal_model=pickle.load(open('model_xgb.pkl','rb'))

#路由主页
@app.route("/")
def homepage():           
    return render_template("homepage.html")

#路由运动推荐
@app.route("/workoutrec",methods=['GET', 'POST'])
def workoutRec():       
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
    
    session['user_input_calories']=calories

    rf.load_data_from_path('./testdata.csv')
    rf.load_model_from_path('./model_run.m', './model_bike.m', './model_mountain.m')
    if session.get('user'):
        data=rf.predict_data(session.get('userId'), calories)
    else:
        data=rf.predict_data(1, calories)
        #flash 一个新用户的话
    
    #todo动态推荐

    run_time,bike_time,mbike_time=readsplitdata(data)
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
    count,re,re_breakfast,re_lunch,re_dinner,re_dessert=0,0,0,0,0,0
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
    #load DietRec model
    diet_data = dietRec.recipe_rec(calories, count,s_breakfast, s_lunch,
    s_dinner, s_dessert, s_vegan, re,re_breakfast,re_lunch,re_dinner,re_dessert)

    #save DietRec parameter into global variate
    diet_list=calories, count,s_breakfast, s_lunch,s_dinner, s_dessert, s_vegan, re,re_breakfast,re_lunch,re_dinner,re_dessert
    session['diet_list']=diet_list

    meal_type=diet_data.get('Meal_Type')
    breakfast_num,lunch_num,dinner_num,dessert_num=splitMeal(meal_type)

    if breakfast_num != 0:
        flash('Breakfast')
        df_breakfast= generateBreakfastDataFrame(diet_data,breakfast_num)
        # use pandas method to auto generate html
        df_html_b = df_breakfast.T.to_html(classes="table_rec",header=False,index=False) 
        label_tbreakfast = 'Recommonded Breakfast'
    else:
        df_html_b=''
        label_tbreakfast=''

    if lunch_num !=0:
        flash('Lunch')
        df_lunch=generateLunchDataFrame(diet_data,breakfast_num,lunch_num)
        df_html_l = df_lunch.T.to_html(classes="table_rec",header=False,index=False) 
        label_tlunch='Recommonded Lunch'
    else:
        df_html_l=''
        label_tlunch=''

    if dinner_num !=0:
        flash('Dinner')
        df_dinner=generateDinnerDataFrame(diet_data,breakfast_num,lunch_num,dinner_num)
        df_html_dinner = df_dinner.T.to_html(classes="table_rec",header=False,index=False) 
        label_tdinner='Recommonded Dinner'
    else:
        df_html_dinner=''
        label_tdinner=''
        
    if dessert_num !=0:   
        flash('Dessert')
        df_dessert=generateDessertDataFrame(diet_data,breakfast_num,lunch_num,dinner_num,dessert_num)
        df_html_dessert = df_dessert.T.to_html(classes="table_rec",header=False,index=False) 
        label_tdessert='Recommonded Dessert'
    else:
        df_html_dessert=''
        label_tdessert=''

    data_img_urls=diet_data.get('img_urls') 
    
#re增加时，其他都初始化为0
    return render_template("dietrec_result.html",table_b_html=df_html_b,
    table_l_html=df_html_l,table_dinner_html=df_html_dinner,table_dessert_html=df_html_dessert,
    label_tbreakfast=label_tbreakfast,label_tlunch=label_tlunch,label_tdinner=label_tdinner,
    label_tdessert=label_tdessert)

def append_list(header,input_list,start,end):
    for i in range(start,end):
        header.append(input_list[i])
    return header

def append_url(header,input_list,start,end):
    for i in range(start,end):
        header.append(input_list[i])
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
def generateBreakfastDataFrame(diet_data,breakfast_num):
    
    data_url=['Image']
    # url=''.join(diet_data.get('img_urls'))
    # print(url)
    # urls = url.split(',')
    # data_url.append(urls[0])
    append_list(data_url,diet_data.get('img_urls'),0,breakfast_num)

    data_name=["Meal"]
    append_list(data_name,diet_data.get('Name'),0,breakfast_num)

    data_Ingredients_list=["Ingredients List"]
    append_list(data_Ingredients_list,diet_data.get('Ingredients_list'),0,breakfast_num)

    data_cal=["Meal Calories"]
    append_list(data_cal,diet_data.get('Calorie_num'),0,breakfast_num)

    df = pd.DataFrame(data=[data_url,data_name,data_cal,data_Ingredients_list])
    return df
def generateLunchDataFrame(diet_data,breakfast_num,lunch_num):
    data_name=["Meal"]
    append_list(data_name,diet_data.get('Name'),breakfast_num,breakfast_num+lunch_num)

    data_Ingredients_list=["Ingredients List"]
    append_list(data_Ingredients_list,diet_data.get('Ingredients_list'),breakfast_num,breakfast_num+lunch_num)

    data_cal=["Meal Calories"]
    append_list(data_cal,diet_data.get('Calorie_num'),breakfast_num,breakfast_num+lunch_num)

    df = pd.DataFrame(data=[data_name,data_cal,data_Ingredients_list])
    return df   
def generateDinnerDataFrame(diet_data,breakfast_num,lunch_num,dinner_num):
    data_name=["Meal"]
    append_list(data_name,diet_data.get('Name'),breakfast_num+lunch_num,breakfast_num+lunch_num+dinner_num)

    data_Ingredients_list=["Ingredients List"]
    append_list(data_Ingredients_list,diet_data.get('Ingredients_list'),breakfast_num+lunch_num,breakfast_num+lunch_num+dinner_num)

    data_cal=["Meal Calories"]
    append_list(data_cal,diet_data.get('Calorie_num'),breakfast_num+lunch_num,breakfast_num+lunch_num+dinner_num)

    df = pd.DataFrame(data=[data_name,data_cal,data_Ingredients_list])
    return df
def generateDessertDataFrame(diet_data,breakfast_num,lunch_num,dinner_num,dessert_num):
    data_name=["Meal"]
    append_list(data_name,diet_data.get('Name'),breakfast_num+lunch_num+dinner_num,breakfast_num+lunch_num+dinner_num+dessert_num)

    data_Ingredients_list=["Ingredients List"]
    append_list(data_Ingredients_list,diet_data.get('Ingredients_list'),breakfast_num+lunch_num+dinner_num,breakfast_num+lunch_num+dinner_num+dessert_num)

    data_cal=["Meal Calories"]
    append_list(data_cal,diet_data.get('Calorie_num'),breakfast_num+lunch_num+dinner_num,breakfast_num+lunch_num+dinner_num+dessert_num)

    df = pd.DataFrame(data=[data_name,data_cal,data_Ingredients_list])
    return df

#   'img_urls': ['https://images.media-allrecipes.com/userphotos/125x70/1110710.jpg , https://images.media-allrecipes.com/userphotos/560x315/1110710.jpg , ', 'https://images.media-allrecipes.com/userphotos/560x315/819709.jpg , https://images.media-allrecipes.com/userphotos/125x70/819709.jpg , https://images.media-allrecipes.com/userphotos/125x70/7079477.jpg , https://images.media-allrecipes.com/userphotos/125x70/7079476.jpg , https://images.media-allrecipes.com/userphotos/125x70/3083932.jpg , https://images.media-allrecipes.com/userphotos/125x70/2209681.jpg , https://images.media-allrecipes.com/userphotos/125x70/1120488.jpg , '], 'Meal_Type': ['breakfast', 'lunch'], 'veg': ['vegetarian', 'vegetarian']}

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

    meal_type=diet_data.get('Meal_Type')
    breakfast_num,lunch_num,dinner_num,dessert_num=splitMeal(meal_type)

    if breakfast_num != 0:
        flash('Breakfast')
        df_breakfast= generateBreakfastDataFrame(diet_data,breakfast_num)
        # use pandas method to auto generate html
        df_html_b = df_breakfast.T.to_html(classes="table_rec",header=False,index=False) 
        label_tbreakfast = 'Recommonded Breakfast'
    else:
        df_html_b=''
        label_tbreakfast=''

    if lunch_num !=0:
        flash('Lunch')
        df_lunch=generateLunchDataFrame(diet_data,breakfast_num,lunch_num)
        df_html_l = df_lunch.T.to_html(classes="table_rec",header=False,index=False) 
        label_tlunch='Recommonded Lunch'
    else:
        df_html_l=''
        label_tlunch=''

    if dinner_num !=0:
        flash('Dinner')
        df_dinner=generateDinnerDataFrame(diet_data,breakfast_num,lunch_num,dinner_num)
        df_html_dinner = df_dinner.T.to_html(classes="table_rec",header=False,index=False) 
        label_tdinner='Recommonded Dinner'
    else:
        df_html_dinner=''
        label_tdinner=''
        
    if dessert_num !=0:   
        flash('Dessert')
        df_dessert=generateDessertDataFrame(diet_data,breakfast_num,lunch_num,dinner_num,dessert_num)
        df_html_dessert = df_dessert.T.to_html(classes="table_rec",header=False,index=False) 
        label_tdessert='Recommonded Dessert'
    else:
        df_html_dessert=''
        label_tdessert=''

    data_img_urls=diet_data.get('img_urls') 
    
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

    meal_type=diet_data.get('Meal_Type')
    breakfast_num,lunch_num,dinner_num,dessert_num=splitMeal(meal_type)

    if breakfast_num != 0:
        flash('Breakfast')
        df_breakfast= generateBreakfastDataFrame(diet_data,breakfast_num)
        # use pandas method to auto generate html
        df_html_b = df_breakfast.T.to_html(classes="table_rec",header=False,index=False) 
        label_tbreakfast = 'Recommonded Breakfast'
    else:
        df_html_b=''
        label_tbreakfast=''

    if lunch_num !=0:
        flash('Lunch')
        df_lunch=generateLunchDataFrame(diet_data,breakfast_num,lunch_num)
        df_html_l = df_lunch.T.to_html(classes="table_rec",header=False,index=False) 
        label_tlunch='Recommonded Lunch'
    else:
        df_html_l=''
        label_tlunch=''

    if dinner_num !=0:
        flash('Dinner')
        df_dinner=generateDinnerDataFrame(diet_data,breakfast_num,lunch_num,dinner_num)
        df_html_dinner = df_dinner.T.to_html(classes="table_rec",header=False,index=False) 
        label_tdinner='Recommonded Dinner'
    else:
        df_html_dinner=''
        label_tdinner=''
        
    if dessert_num !=0:   
        flash('Dessert')
        df_dessert=generateDessertDataFrame(diet_data,breakfast_num,lunch_num,dinner_num,dessert_num)
        df_html_dessert = df_dessert.T.to_html(classes="table_rec",header=False,index=False) 
        label_tdessert='Recommonded Dessert'
    else:
        df_html_dessert=''
        label_tdessert=''

    data_img_urls=diet_data.get('img_urls') 
    
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

    meal_type=diet_data.get('Meal_Type')
    breakfast_num,lunch_num,dinner_num,dessert_num=splitMeal(meal_type)

    if breakfast_num != 0:
        flash('Breakfast')
        df_breakfast= generateBreakfastDataFrame(diet_data,breakfast_num)
        # use pandas method to auto generate html
        df_html_b = df_breakfast.T.to_html(classes="table_rec",header=False,index=False) 
        label_tbreakfast = 'Recommonded Breakfast'
    else:
        df_html_b=''
        label_tbreakfast=''

    if lunch_num !=0:
        flash('Lunch')
        df_lunch=generateLunchDataFrame(diet_data,breakfast_num,lunch_num)
        df_html_l = df_lunch.T.to_html(classes="table_rec",header=False,index=False) 
        label_tlunch='Recommonded Lunch'
    else:
        df_html_l=''
        label_tlunch=''

    if dinner_num !=0:
        flash('Dinner')
        df_dinner=generateDinnerDataFrame(diet_data,breakfast_num,lunch_num,dinner_num)
        df_html_dinner = df_dinner.T.to_html(classes="table_rec",header=False,index=False) 
        label_tdinner='Recommonded Dinner'
    else:
        df_html_dinner=''
        label_tdinner=''
        
    if dessert_num !=0:  
        flash('Dessert') 
        df_dessert=generateDessertDataFrame(diet_data,breakfast_num,lunch_num,dinner_num,dessert_num)
        df_html_dessert = df_dessert.T.to_html(classes="table_rec",header=False,index=False) 
        label_tdessert='Recommonded Dessert'
    else:
        df_html_dessert=''
        label_tdessert=''

    data_img_urls=diet_data.get('img_urls') 
    
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

    meal_type=diet_data.get('Meal_Type')
    breakfast_num,lunch_num,dinner_num,dessert_num=splitMeal(meal_type)

    if breakfast_num != 0:
        flash('Breakfast')
        df_breakfast= generateBreakfastDataFrame(diet_data,breakfast_num)
        # use pandas method to auto generate html
        df_html_b = df_breakfast.T.to_html(classes="table_rec",header=False,index=False) 
        label_tbreakfast = 'Recommonded Breakfast'
    else:
        df_html_b=''
        label_tbreakfast=''

    if lunch_num !=0:
        flash('Lunch')
        df_lunch=generateLunchDataFrame(diet_data,breakfast_num,lunch_num)
        df_html_l = df_lunch.T.to_html(classes="table_rec",header=False,index=False) 
        label_tlunch='Recommonded Lunch'
    else:
        df_html_l=''
        label_tlunch=''

    if dinner_num !=0:
        flash('Dinner')
        df_dinner=generateDinnerDataFrame(diet_data,breakfast_num,lunch_num,dinner_num)
        df_html_dinner = df_dinner.T.to_html(classes="table_rec",header=False,index=False) 
        label_tdinner='Recommonded Dinner'
    else:
        df_html_dinner=''
        label_tdinner=''
        
    if dessert_num !=0:   
        flash('Dessert')
        df_dessert=generateDessertDataFrame(diet_data,breakfast_num,lunch_num,dinner_num,dessert_num)
        df_html_dessert = df_dessert.T.to_html(classes="table_rec",header=False,index=False) 
        label_tdessert='Recommonded Dessert'
    else:
        df_html_dessert=''
        label_tdessert=''

    data_img_urls=diet_data.get('img_urls') 
    
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

    if breakfast_num != 0:
        flash('Breakfast')
        df_breakfast= generateBreakfastDataFrame(diet_data,breakfast_num)
        # use pandas method to auto generate html
        df_html_b = df_breakfast.T.to_html(classes="table_rec",header=False,index=False) 
        label_tbreakfast = 'Recommonded Breakfast'
    else:
        df_html_b=''
        label_tbreakfast=''

    if lunch_num !=0:
        flash('Lunch')
        df_lunch=generateLunchDataFrame(diet_data,breakfast_num,lunch_num)
        df_html_l = df_lunch.T.to_html(classes="table_rec",header=False,index=False) 
        label_tlunch='Recommonded Lunch'
    else:
        df_html_l=''
        label_tlunch=''

    if dinner_num !=0:
        flash('Dinner')
        df_dinner=generateDinnerDataFrame(diet_data,breakfast_num,lunch_num,dinner_num)
        df_html_dinner = df_dinner.T.to_html(classes="table_rec",header=False,index=False) 
        label_tdinner='Recommonded Dinner'
    else:
        df_html_dinner=''
        label_tdinner=''
        
    if dessert_num !=0:  
        flash('Dessert') 
        df_dessert=generateDessertDataFrame(diet_data,breakfast_num,lunch_num,dinner_num,dessert_num)
        df_html_dessert = df_dessert.T.to_html(classes="table_rec",header=False,index=False) 
        label_tdessert='Recommonded Dessert'
    else:
        df_html_dessert=''
        label_tdessert=''

    data_img_urls=diet_data.get('img_urls') 
    
    return render_template("dietrec_result.html",table_b_html=df_html_b,
    table_l_html=df_html_l,table_dinner_html=df_html_dinner,table_dessert_html=df_html_dessert,
    label_tbreakfast=label_tbreakfast,label_tlunch=label_tlunch,label_tdinner=label_tdinner,
    label_tdessert=label_tdessert)

#路由运动记录    
@app.route("/activitylog")
def activitylog(): 
    if session.get('user'):

        calories = int(session.get('user_input_calories'))
        #get mock data
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
        duration=cal_time(duration_seconds)
        sport_type=check_sport_type(bike_check,mbike_check,run_check)
        
        print(distance)
        system_time=str(datetime.now())
        time=system_time.split('.')[0]
        print(time)
        #mike model
        HR_track_model = torch.load('./model_heartrate_01.pt', map_location=torch.device('cpu'))
        use_cuda=torch.cuda.is_available() 
        HR_output = HR_track_model.predict()
        output_pre, output_tar, x = '', '', "''"
        for v in HR_output[0][0]:
            output_pre = output_pre + str(v) + ','
        for v in HR_output[0][1]:
            output_tar = output_tar + str(v) + ','
        output_pre, output_tar = output_pre[:-1], output_tar[:-1]
        for i in range(50):
            x = x + ", \"\""
       
        #oni model
        acc_output = calories_cal_model.predict(input_data.iloc[:1])
        print(acc_output)
        actual_calories = int(acc_output)
        return render_template("activitylog.html",actual_calories=actual_calories,
        time=time,distance=distance,sport_type=sport_type,duration=duration,avg_speed=avg_speed,
        avg_heart_rate=avg_heart_rate, heartrate_pre=output_pre, heartrate_tar=output_tar, 
        xaxis=x)
    else:
        flash('只有登录后才能使用 //todo')
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
        db = pymysql.connect(host="localhost",user="root",password="961214",database="Fitastic")
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
        db = pymysql.connect(host="localhost",user="root",password="961214",database="Fitastic")
        cursor = db.cursor()
        username = request.form.get('username', None)
        email = request.form.get('email', None)
        password = request.form.get('password', None)
        repassword = request.form.get('repassword', None)
        sql = "select * from users where username= '{}'".format(username, encoding='utf-8')
        try:
            cursor.execute(sql)
            result = cursor.fetchone()
            if result == None:
                sql = 'INSERT INTO users (username, password, email) VALUES ("{}", "{}", "{}")'.format(username,md5(password.encode('utf-8')).hexdigest(), email, encoding='utf-8')
                print (sql)
                cursor.execute(sql)
                flash("registered successfully, please sign in")
                return redirect(url_for('login'))
            else:
                if result[0][2] == md5(password.encode('utf-8')).hexdigest():
                    flash("please sign up with another name") 
                    return redirect(url_for('signup'))
                if password != repassword:
                    flash("Inconsistency of password!")
                    return redirect(url_for('signup'))
                db.commit()
        except:
            # 如果发生错误则回滚
            traceback.print_exc()
            db.rollback()
        # 关闭数据库连接
        db.close()
    return render_template("signup.html") 


@app.route("/reset", methods=['GET', 'POST'])
def reset():
    if request.method == 'POST':     
        db = pymysql.connect(host="localhost",user="root",password="961214",database="Fitastic") #换成自己的root和password
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
    