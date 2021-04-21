# Fitastic System
## Description
Fitastic is a LSTM based system that ultilizes the personal workout records with typical feature patterns to recommend a customized workout and a diet plan. Instead of collecting users' height, weight, and other private information, Fitastic regards users’ heart rate changes as an important indicator to establish model. Calorie is an important input number for Fitastic to understand how much calorie users want to consume and take in, and recommend plans on that basis.

## Functions
* To provide personal workout recommendation (Run, Bike, MountainBike) based on input calorie
* To provide heart rate prediction during workout, and real-time suggestions on speeding up or slowing down for reaching the target calorie consumption
* To make actual calorie consumption calculation after workout, and generate activity report
* To provide personal diet recommendation based on input calorie

## Data
* Over 50 thousand workout records of over 900 users collected from [https://www.endomondo.com/](https://www.endomondo.com/)
* Almost 20 thousand recipes scraped from [https://www.allrecipes.com/](https://www.allrecipes.com/)
