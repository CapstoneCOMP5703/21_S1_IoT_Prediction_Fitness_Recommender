import pandas as pd
import numpy as np

# display the recipe details
class DetailsDisplay:
    def details_display(self, recipe_name):
        recipe = pd.read_csv("./dataset/recipes.csv")
        
        ingredient_list = recipe.Ingredients_list[recipe.Name==recipe_name]
        direction_list = recipe.Directions_list[recipe.Name==recipe_name]
        img_url = recipe.img_urls[recipe.Name==recipe_name]
        calorie = recipe.Calorie_num[recipe.Name==recipe_name]
        prep_time = recipe.Prep_time[recipe.Name==recipe_name]
        cook_time = recipe.Cook_time[recipe.Name==recipe_name]
        meal_type = recipe.veg[recipe.Name==recipe_name]
        index_value = recipe.index.values[recipe.Name==recipe_name]
        
        return ingredient_list,direction_list,img_url,calorie,prep_time,cook_time,meal_type,index_value
