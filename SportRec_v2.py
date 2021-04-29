import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import joblib

# from sklearn import preprocessing
# lbl = preprocessing.LabelEncoder()
# train_x['calories'] = lbl.fit_transform(train_x['calories'].astype(str))

class Model:
    def __init__(self):
        self.gender_dict = {'female': 0, 'male': 1, 'unknown': 2}
        self.data_ready = 0
        self.model_ready = 0

    def load_data_from_path(self, test_data_path):
        
        self.data_ready = 1
        self.data = pd.read_csv(test_data_path)


    def load_model_from_path(self, model_run_path, model_bike_path, model_mountion_path):
        # load model
        self.model_ready = 1
        self.run_clf = joblib.load(model_run_path)
        self.bike_clf = joblib.load(model_bike_path)
        self.mountain_clf = joblib.load(model_mountion_path)

    def load_model_from_sklearn(self, model_run, model_bike, model_mountion):
        self.model_ready = 1
        self.run_clf = model_run
        self.bike_clf = model_bike
        self.mountain_clf = model_mountion

    def predict_data(self, userId, calories):
        gender_dict = {'female': 0, 'male': 1, 'unknown': 2}
        if self.data_ready == 0:
            print('Please load data!')
        if self.model_ready == 0:
            print('Please load model!')
        self.test_df = self.data.copy()
        self.test_df['gender_idx'] = self.test_df.gender.apply(lambda x: gender_dict[x])
        tmp_df = self.test_df[self.test_df.userId == userId]
        if len(tmp_df) == 0:  # identify new users
            run_data = pd.DataFrame([[2, calories, 155, 11]], columns=['gender_idx', 'calories', 'avg_heart_rate', 'avg_speed'])
            run_res = self.run_clf.predict(run_data)[0]
            bike_data = pd.DataFrame([[2, calories, 145, 26]], columns=['gender_idx', 'calories', 'avg_heart_rate', 'avg_speed'])
            bike_res = self.bike_clf.predict(bike_data)[0]
            mountain_data = pd.DataFrame([[2, calories, 150, 20]], columns=['gender_idx', 'calories', 'avg_heart_rate', 'avg_speed'])
            mountain_res = self.mountain_clf.predict(mountain_data)[0]
        else:
            try:
                # according to ID index users' records
                run_data = self.test_df[(self.test_df.userId == userId) & (self.test_df.sport == 'run')]
                # extract features
                run_data = run_data[['gender_idx', 'calories', 'avg_heart_rate', 'avg_speed']]
                # define calories
                run_data['calories'] = calories
                # predict historical records
                run_pred_list = self.run_clf.predict(run_data)
                # output result
                run_res = np.mean(run_pred_list)
            except:
                run_res = -1

            try:
                bike_data = self.test_df[(self.test_df.userId == userId) & (self.test_df.sport == 'bike')]
                bike_data = bike_data[['gender_idx', 'calories', 'avg_heart_rate', 'avg_speed']]
                bike_data['calories'] = calories
                bike_pred_list = self.bike_clf.predict(bike_data)
                bike_res = np.mean(bike_pred_list)
            except:
                bike_res = -1

            try:
                mountain_data = self.test_df[(self.test_df.userId == userId) & (self.test_df.sport == 'mountain bike')]
                mountain_data = mountain_data[['gender_idx', 'calories', 'avg_heart_rate', 'avg_speed']]
                mountain_data['calories'] = calories
                mountain_pred_list = self.mountain_clf.predict(mountain_data)
                mountain_res = np.mean(mountain_pred_list)
            except:
                mountain_res = -1

        res_dict = {'run': run_res, 'bike': bike_res, 'mountainbike': mountain_res}

        print_list = []
        for k, v in res_dict.items():
            if v >= 0:
                print_list.append('{} : {}'.format(k, v // 60))
            # print(print_list)
        return print_list
    
if __name__ == '__main__':
    rf = Model()
    # load data
    rf.load_data_from_path('./testdata.csv')
    # load model
    rf.load_model_from_path('./model_run.m', './model_bike.m', './model_mountain.m')
    # predict 
    data = rf.predict_data(11111116, 400)
    # print
    print(data)

