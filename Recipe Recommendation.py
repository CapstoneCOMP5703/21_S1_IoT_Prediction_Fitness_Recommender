#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import re
from nltk.stem import WordNetLemmatizer


# In[ ]:


def remove(x):
    x = re.findall("'(.*?)'",x)
    x = [el for el in x if el != '' and el != 'Add all ingredients to list'] 
    return str(x)


# In[ ]:


#data preprocessing
def preprocessing():
    #read data
    recipe = pd.read_csv("/Users/apple/Downloads/recipe_dataset.csv")
    #remove null values
    recipe.dropna(axis=0, how="any", inplace=True)
    #split calorie
    recipe["Calorie_num"]=recipe["Calorie"].map(lambda x:int(x.split(" ")[0]))
    #remove the duplicates
    recipe.drop_duplicates(subset=["Name"], keep="first", inplace=True)
    #reset the index
    recipe = recipe.reset_index(drop=True)
    #remove the null value & useless value of the ingredient list
    recipe['Ingredients_list'] = recipe['Ingredients_list'].apply(remove)
    #Lemmatize the ingredients field
    recipe["lemmatized_ingredient"] = ["".join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]) for lists in recipe['Ingredients']]
    #Adding Veg/Non-Veg information
    mylist =['rib','crab','prosciutto','lobster','sausage','clam','goose','fish','goat','chicken','beef','pork','prawn','egg','Katsuobushi','mackrel','fillet','lamb','steak','salmon','shrimp','bacon','ham','turkey','duck','seafood','squid']
    pattern = '|'.join(mylist)
    recipe["veg"]=recipe.lemmatized_ingredient.str.contains(pattern) 
    recipe.loc[recipe.veg == True,"veg"] = 'non-vegetarian'
    recipe.loc[recipe.veg == False,"veg"] = 'vegetarian'
    return recipe


# In[ ]:


#find the closet row
def find_close(arr, value, meal_type, vegan):
    if vegan == "Yes":
        all_index = np.where((recipe["Meal_Type"]==meal_type) & (recipe["veg"]=="vegetarian"))[0]
    else:
        all_index = np.where(recipe["Meal_Type"]==meal_type)[0]   
    index = np.abs(arr-value).argmin()
    return all_index[index]


# In[ ]:


#print the recommendation list
def print_list(arr1, arr2, calorie, min_calorie, meal_type, vegan):
    while calorie >= min_calorie:
        if vegan == "Yes":
            index = find_close(arr1.loc[(recipe["Meal_Type"]==meal_type) & (recipe["veg"] == "vegetarian")]["Calorie_num"], calorie, meal_type, vegan)
        else:
            index = find_close(arr1.loc[recipe["Meal_Type"]==meal_type]["Calorie_num"], calorie, meal_type, vegan)
        calorie = calorie-arr1["Calorie_num"][index]
        for key in arr2:
            arr2[key].append(arr1[key][index])
    return arr2


# In[ ]:


#calculate the remaining calorie
def cal_calorie_left(dic,calorie):
    num = 0
    for i in dic["Calorie_num"]:
        num = num + i
    return calorie-num


# In[ ]:


#minimal calorie for each meal type
def min_calorie(meal_type, vegan):
    if vegan == "Yes":
        min_calorie = min(recipe.loc[(recipe["Meal_Type"]==meal_type) & (recipe["veg"] == "vegetarian")]["Calorie_num"])
    else:
        min_calorie = min(recipe.loc[recipe["Meal_Type"]==meal_type]["Calorie_num"])  
    return min_calorie


# In[ ]:


def recipe_rec(calorie, meal_num):
    recipe = preprocessing()
    
    #the output for each meal type
    rec_list={"Name":[],"Ingredients_list":[],"Directions_list":[],"Prep_time":[],"Cook_time":[],"Calorie_num":[],"img_urls":[],"Meal_Type":[],"veg":[]}
    
    #limit the number of meal in 1 to 4, and the input calories has to be over 0
    if calorie >= 0: 
        if meal_num == 1:
            #default meal type is breakfast
            meal_type = input("Preferred meal-breakfast,lunch,dinner,dessert? ")
            meal_pre = input("If vegetarian? Yes/No: ")

            print(print_list(recipe,rec_list,calorie,min_calorie(meal_type,meal_pre),meal_type,meal_pre))
                
        elif meal_num == 2:
            #default meal types are breakfast and lunch
            meal_type_one = input("Preferred meal combination? First choice: ")
            meal_type_two = input("Preferred meal combination? Second choice: ")
            meal_pre = input("If vegetarian? Yes/No: ")

            calorie_left = cal_calorie_left(print_list(recipe,rec_list,calorie*0.5,min_calorie(meal_type_one, meal_pre),meal_type_one, meal_pre),calorie)
            print_list(recipe,rec_list,calorie_left,min_calorie(meal_type_two, meal_pre),meal_type_two, meal_pre)
            print(rec_list)
            
        elif meal_num == 3:
            #default meal types are breakfast, lunch & dinner
            meal_type_one = input("Preferred meal combination? First choice: ")
            meal_type_two = input("Preferred meal combination? Second choice: ")
            meal_type_three = input("Preferred meal combination? Third choice: ")
            meal_pre = input("If vegetarian? Yes/No: ")

            calorie_left = cal_calorie_left(print_list(recipe,rec_list,calorie*0.4,min_calorie(meal_type_one,meal_pre),meal_type_one, meal_pre),calorie)
            calorie_left = cal_calorie_left(print_list(recipe,rec_list,calorie*0.4,min_calorie(meal_type_two,meal_pre),meal_type_two, meal_pre),calorie_left)
            print_list(recipe,rec_list,calorie_left,min_calorie(meal_type_three,meal_pre),meal_type_three,meal_pre)    
            print(rec_list)
            
        elif meal_num == 4:
            meal_type_list=["breakfast","lunch","dinner"]
            meal_pre = input("If vegetarian? Yes/No: ")
            calorie_left = calorie
            for types in meal_type_list:
                calorie_left = cal_calorie_left(print_list(recipe,rec_list,calorie*0.3,min_calorie(types,meal_pre),types,meal_pre), calorie_left)
            rec_lists = print_list(recipe,rec_list,calorie_left,min_calorie("dessert",meal_pre),"dessert",meal_pre)
            print(rec_list)
        else:
            print("The number of meal has to be in 1 to 4.")
    else:
        print("The calorie number has be over 0")


# In[ ]:


calorie = int(input("Please input a calorie intake: "))
meal_num = int(input("Please input the number of meal: "))
recipe_rec(calorie, meal_num)


# In[ ]:




