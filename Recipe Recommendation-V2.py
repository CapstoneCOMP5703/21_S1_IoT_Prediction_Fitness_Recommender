#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


class DietRec():
    #find the closet row
    def find_close(self, arr, value, meal_type, vegan, index_number):
        if vegan == 1:
            all_index = np.where((arr["Meal_Type"]==meal_type) & (arr["veg"]=="vegetarian"))[0]
            calories = arr.loc[(arr["Meal_Type"]==meal_type) & (arr["veg"] == "vegetarian")]["Calorie_num"]
        else:
            all_index = np.where(arr["Meal_Type"]==meal_type)[0] 
            calories = arr.loc[arr["Meal_Type"]==meal_type]["Calorie_num"]
        index = list(np.abs(calories-value).sort_values().index)
        return index[index_number]
    
    #calculate the remaining calorie
    def cal_calorie_left(self, dic, calorie):
        num = 0
        for i in dic["Calorie_num"]:
            num = num + i
        return calorie-num
    
    #minimal calorie for each meal type
    def min_calorie(self, arr, meal_type, vegan):
        if vegan == 1:
            min_calorie = min(arr.loc[(arr["Meal_Type"]==meal_type) & (arr["veg"] == "vegetarian")]["Calorie_num"])
        else:
            min_calorie = min(arr.loc[arr["Meal_Type"]==meal_type]["Calorie_num"])  
        return min_calorie
    
    #print the recommendation list
    def print_list(self, arr1, arr2, calorie, min_calorie, meal_type, vegan, index_number):
        while calorie >= min_calorie:
            index = self.find_close(arr1, calorie, meal_type, vegan, index_number)
            calorie = calorie-arr1["Calorie_num"][index]
            for key in arr2:
                arr2[key].append(arr1[key][index])
        return arr2
    
    #main function
    def recipe_rec(self, calorie, meal_num, breakfast, lunch, dinner, dessert, vegan, index_number):
        
        recipe = pd.read_csv("/Users/apple/Downloads/recipes.csv")

        #determine the meal type(s)
        meal_type_dic = {"breakfast":breakfast, "lunch":lunch, "dinner":dinner, "dessert":dessert} 
        meal_type_list = []
        for key,value in meal_type_dic.items():
            if value == 1:
                meal_type_list.append(key)

        #determine the vegan/non-vegan
        meal_pre = vegan
        #the output for each meal type
        rec_list={"Name":[],"Ingredients_list":[],"Directions_list":[],"Prep_time":[],"Cook_time":[],"Calorie_num":[],"img_urls":[],"Meal_Type":[],"veg":[]}

        if meal_num == 1:
            print(type(recipe))
            return self.print_list(recipe,rec_list,calorie,self.min_calorie(recipe,meal_type_list[0],meal_pre),meal_type_list[0],meal_pre,index_number)

        elif meal_num == 2:           
            calorie_left = self.cal_calorie_left(self.print_list(recipe,rec_list,calorie*0.5,self.min_calorie(recipe, meal_type_list[0], meal_pre),meal_type_list[0], meal_pre, index_number),calorie)
            self.print_list(recipe,rec_list,calorie_left,self.min_calorie(meal_type_list[1], meal_pre),meal_type_list[1], meal_pre)
            return rec_list 

        elif meal_num == 3:
            calorie_left = self.cal_calorie_left(self.print_list(recipe,rec_list,calorie*0.4,self.min_calorie(recipe, meal_type_list[0], meal_pre),meal_type_list[0], meal_pre, index_number),calorie)
            calorie_left = self.cal_calorie_left(self.print_list(recipe,rec_list,calorie*0.4,min_calorie(recipe, meal_type_list[1], meal_pre),meal_type_list[1], meal_pre, index_number),calorie)
            self.print_list(recipe,rec_list,calorie_left,self.min_calorie(recipe, meal_type_list[2], meal_pre),meal_type_list[2],meal_pre, index_number)    
            return rec_list

        elif meal_num == 4:
            calorie_left = self.cal_calorie_left(self.print_list(recipe,rec_list,calorie*0.3,self.min_calorie(recipe, meal_type_list[0], meal_pre), meal_type_list[0], meal_pre, index_number), calorie)
            calorie_left = self.cal_calorie_left(self.print_list(recipe,rec_list,calorie*0.3,self.min_calorie(recipe, meal_type_list[1], meal_pre), meal_type_list[1], meal_pre, index_number), calorie)
            calorie_left = self.cal_calorie_left(self.print_list(recipe,rec_list,calorie*0.3,self.min_calorie(recipe, meal_type_list[2], meal_pre), meal_type_list[2], meal_pre, index_number), calorie)
            rec_lists = self.print_list(recipe,rec_list,calorie_left,self.min_calorie(recipe, meal_type_list[3],meal_pre),meal_type_list[3],meal_pre, index_number)
            return rec_list

if __name__ == '__main__':
    dietRec = DietRec()
    data = dietRec.recipe_rec(2000,1,0,1,0,0,0,3)
    print(data)


# In[ ]:




