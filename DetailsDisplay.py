import pandas as pd
import numpy as np

class DetailsDisplay:
    def details_display(self, recipe_name):
        recipe = pd.read_csv("/Users/apple/Downloads/recipes.csv")
        
        ingredient_list = recipe.Ingredients_list[recipe.Name==recipe_name]
        direction_list = recipe.Directions_list[recipe.Name==recipe_name]
        img_url = recipe.img_urls[recipe.Name==recipe_name]
        calorie = recipe.Calorie_num[recipe.Name==recipe_name]
        prep_time = recipe.Prep_time[recipe.Name==recipe_name]
        cook_time = recipe.Cook_time[recipe.Name==recipe_name]
        meal_type = recipe.veg[recipe.Name==recipe_name]
        
        return ingredient_list,direction_list,img_url,calorie,prep_time,cook_time,meal_type

if __name__ == '__main__':
    detailsDisplay = DetailsDisplay()
    ingredient_list,direction_list,img_url,calorie,prep_time,cook_time,meal_type = detailsDisplay.details_display("easy morning glory muffins")
    print(ingredient_list[0],direction_list[0],img_url[0],calorie[0],prep_time[0],cook_time[0],meal_type[0])