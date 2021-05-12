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

## Steps
### Configuration 1: MySQL installation and import .sql file 
1. Download MySQL (MAC/Windows): [https://dev.mysql.com/downloads/mysql/](https://dev.mysql.com/downloads/mysql/)
2. Download users.sql: [https://github.com/CapstoneCOMP5703/CS25-2/blob/main/users.sql](https://github.com/CapstoneCOMP5703/CS25-2/blob/main/users.sql)
3. Put the users.sql in the MySQL bin directory
4. Open Terminal (cmd)
5. Run command `mysqld install` and you will get following message `Service successfully installed.` 
6. Run command `net start mysql`
7. Set username and password, run command: `mysqladmin -u your_username -p password`
8. Set the password and confirm it
9. Run command `mysql -u your_username -p` and enter your password
10. Run command `create database Fitastic;`
11. Run command `use Fitastic;` and `source users.sql`

### Configuration 2: Change sql configuration in app.py
1. Open app.py: [https://github.com/CapstoneCOMP5703/CS25-2/blob/main/app.py](https://github.com/CapstoneCOMP5703/CS25-2/blob/main/users.sql)
2. Change all mysql connection: `db = pymysql.connect(host="localhost",user="your_username",password="lyour_password",database="Fitastic")`

## Environment
* python 3.8
* MySQL
