# create Macine learning pipeline
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler


# import libraries for regression model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor,ExtraTreesRegressor
from sklearn.linear_model import (Lasso,ElasticNet,Ridge,PassiveAggressiveRegressor,ARDRegression,RANSACRegressor,
TheilSenRegressor,HuberRegressor,Lars,LassoLars,SGDRegressor,BayesianRidge,LinearRegression,OrthogonalMatchingPursuit)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math



class Scaling:
    # constructor
    def __init__(self, data_train,data_test,pred):
        self.data_train = data_train
        self.data_test = data_test
        self.pred = pred

    def standard_scaler(self):
        scaler=StandardScaler()
        scaler.fit(self.data_train)
        x_train=scaler.transform(self.data_train)
        x_test=scaler.transform(self.data_test)
        pred=scaler.transform(self.pred)
        return x_train,x_test,pred
    
    def MinMax(self):
        minmax=MinMaxScaler()
        minmax.fit(self.data_train)
        x_train=minmax.transform(self.data_train)
        x_test=minmax.transform(self.data_test)
        pred=minmax.transform(self.pred)
        return x_train, x_test, pred

class regressionModel():
    def __init__(self,pred,data_train, data_test, test_size=0.2, random_state=32):
        self.data_train = data_train
        self.data_test = data_test
        self.pred = pred
        self.test_size = test_size
        self.random_state = random_state
        self.model=[LinearRegression(),Ridge(),Lasso(),ElasticNet(),SGDRegressor(),
                    KNeighborsRegressor(),DecisionTreeRegressor(),RandomForestRegressor(),
                    AdaBoostRegressor(),XGBRegressor(),GradientBoostingRegressor(),TheilSenRegressor(),
                    RANSACRegressor(),OrthogonalMatchingPursuit(),LassoLars(),Lars(),HuberRegressor(),SVR(),
                    PassiveAggressiveRegressor(),ARDRegression(),BayesianRidge(),BaggingRegressor(),ExtraTreesRegressor()]
        self.model_name=['Linear Regression','Ridge','Lasso','ElasticNet','SGDRegressor',
                'KNeighborsRegressor','DecisionTreeRegressor','RandomForestRegressor',
                'AdaBoostRegressor','XGBRegressor','GradientBoostingRegressor','TheilSenRegressor',
                'RANSACRegressor','OrthogonalMatchingPursuit','LassoLars','Lars','HuberRegressor','SVR'
                'PassiveAggressiveRegressor','ARDRegression','BayesianRidge','BaggingRegressor','ExtraTreesRegressor']
        self.model_table=[]

    def model_fit(self):
        lst=[]
        for i in range(len(self.model)):
            self.model[i].fit(self.X_train, self.y_train)
            y_pred=self.model[i].predict(self.X_test)
            # model evaluation
            mse=mean_squared_error(self.y_test, y_pred)
            rmse=math.sqrt(mse)
            mae=mean_absolute_error(self.y_test, y_pred)
            r2=r2_score(self.y_test, y_pred)
            lst.append({'model':self.model_name[i],'MSE':mse,'RMSE':rmse,'MAE':mae,'r2':r2})
        self.model_table=pd.DataFrame(lst)
        return self.model_table