# create Macine learning pipeline
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
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
    def __init__(self, data_train):
        self.data_train = data_train

    def standard_scaler(self):
        scaler=StandardScaler()
        scaler.fit(self.data_train)
        x_train=scaler.transform(self.data_train)
        return x_train
    
    def MinMax(self):
        minmax=MinMaxScaler()
        minmax.fit(self.data_train)
        x_train=minmax.transform(self.data_train)
        return x_train



class regressionModel():
    def __init__(self,x,y, test_size=0.2, random_state=32):
        self.x = x
        self.y = y
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
    
    def regressionAccuracy(self,y_test,y_pred,model_name,model_obj):
        # model evaluation
        mse=mean_squared_error(y_test, y_pred)
        rmse=math.sqrt(mse)
        mae=mean_absolute_error(y_test, y_pred)
        r2=r2_score(y_test, y_pred)
        return {'model':model_name,'MSE':mse,'RMSE':rmse,'MAE':mae,'r2':r2,'model_obj':model_obj}

    def model_fit(self):
        # split data into training and testing sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x,self.y, test_size=self.test_size, random_state=self.random_state)

        # scaling data
        self.x_train=Scaling(self.x_train).standard_scaler()
        self.x_test=Scaling(self.x_test).standard_scaler()


        lst=[]
        for i,j in zip(self.model_name,self.model):
            model_obj=j.fit(self.x_train,self.y_train)
            y_pred=model_obj.predict(self.x_test)
            lst.append(self.regressionAccuracy(self.y_test, y_pred,i,model_obj))
        self.model_table=pd.DataFrame(lst)
        return self.model_table
    
    def hyperParameter(self):
        lst=[]
        # linear regression
        lr=LinearRegression()
        cv_sets=ShuffleSplit(random_state=4)
        param= {'fit_intercept': [True, False],
                        'normalize': [True, False],
                        'copy_X': [True, False],
                        'n_jobs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'max_iter': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'tol': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                        'warm_start': [True, False],
                        'l1_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        
        scorer= make_scorer(r2_score)
        lr_obj= RandomizedSearchCV(lr, param, cv=cv_sets, scoring=scorer, n_iter=100, random_state=4)
        lr_obj.fit(self.x_train, self.y_train)
        lst.append(self.regressionAccuracy(self.y_test, lr_obj.predict(self.x_test),'Linear Regression',lr_obj))
        # Ridge
        ridge=Ridge()
        cv_sets=ShuffleSplit(random_state=4)
        param= {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        'fit_intercept': [True, False],
                        'normalize': [True, False],
                        'copy_X': [True, False],
                        'max_iter': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'tol': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                        'warm_start': [True, False],
                        'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

        scorer= make_scorer(r2_score)
        ridge_obj= RandomizedSearchCV(ridge, param, cv=cv_sets, scoring=scorer, n_iter=100, random_state=4)
        ridge_obj.fit(self.x_train, self.y_train)
        lst.append(self.regressionAccuracy(self.y_test, ridge_obj.predict(self.x_test),'Ridge',ridge_obj))
        # Lasso
        lasso=Lasso()
        cv_sets=ShuffleSplit(random_state=4)
        param= {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        'fit_intercept': [True, False],
                        'normalize': [True, False],
                        'copy_X': [True, False],
                        'tol':[0.0001, 0.0001, 0.001, 0.01, 0.1, 1],
                        'selection':['cyclic', 'random'],
                        'positive':[True, False],
                        'max_iter': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'tol': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                        'warm_start': [True, False],
                        'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        scorer= make_scorer(r2_score)
        lasso_obj= RandomizedSearchCV(lasso, param, cv=cv_sets, scoring=scorer, n_iter=100, random_state=4)
        lasso_obj.fit(self.x_train, self.y_train)
        lst.append(self.regressionAccuracy(self.y_test, lasso_obj.predict(self.x_test),'Lasso',lasso_obj))
        # elastic net
        elastic=ElasticNet()
        cv_sets=ShuffleSplit(random_state=4)
        param= {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        'l1_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'fit_intercept': [True, False],
                        'normalize': [True, False],
                        'max_iter': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'tol': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                        'selection':['cyclic', 'random'],
                        'positive':[True, False],
                        'warm_start': [True, False],
                        'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        scorer= make_scorer(r2_score)
        elastic_obj= RandomizedSearchCV(elastic, param, cv=cv_sets, scoring=scorer, n_iter=100, random_state=4)
        elastic_obj.fit(self.x_train, self.y_train)
        lst.append(self.regressionAccuracy(self.y_test, elastic_obj.predict(self.x_test),'Elastic Net',elastic_obj))

        # SGDRegressor
        sgd=SGDRegressor()
        cv_sets=ShuffleSplit(random_state=4)
        param= {'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                        'penalty': ['none', 'l2', 'l1', 'elasticnet'],
                        'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
                        'l1_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'fit_intercept': [True, False],
                        'n_iter': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'shuffle': [True, False],
                        'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        scorer= make_scorer(r2_score)
        sgd_obj= RandomizedSearchCV(sgd, param, cv=cv_sets, scoring=scorer, n_iter=100, random_state=4)
        sgd_obj.fit(self.x_train, self.y_train)
        lst.append(self.regressionAccuracy(self.y_test, sgd_obj.predict(self.x_test),'SGD',sgd_obj))
        # KNN
        knn=KNeighborsRegressor()
        cv_sets=ShuffleSplit(random_state=4)
        param= {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'weights': ['uniform', 'distance'],
                        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                        'leaf_size': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'p': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'metric': ['minkowski', 'euclidean', 'cityblock', 'chebyshev', 'mahalanobis'],
                        'metric_params': [{'p': 1}, {'p': 2}, {'p': 3}, {'p': 4}, {'p': 5}]}
        scorer= make_scorer(r2_score)
        knn_obj= RandomizedSearchCV(knn, param, cv=cv_sets, scoring=scorer, n_iter=100, random_state=4)
        knn_obj.fit(self.x_train, self.y_train)
        lst.append(self.regressionAccuracy(self.y_test, knn_obj.predict(self.x_test),'KNN',knn_obj))
        # Decision Tree
        tree=DecisionTreeRegressor()
        cv_sets=ShuffleSplit(random_state=4)
        param= {'criterion': ['mse', 'friedman_mse', 'mae'],
                        'splitter': ['best', 'random'],
                        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'max_features': ['auto', 'sqrt', 'log2', None],
                        'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        scorer= make_scorer(r2_score)
        tree_obj= RandomizedSearchCV(tree, param, cv=cv_sets, scoring=scorer, n_iter=100, random_state=4)
        tree_obj.fit(self.x_train, self.y_train)
        lst.append(self.regressionAccuracy(self.y_test, tree_obj.predict(self.x_test),'Decision Tree',tree_obj))
        # Random Forest
        forest=RandomForestRegressor()
        cv_sets=ShuffleSplit(random_state=4)
        param= {'n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'criterion': ['mse', 'friedman_mse', 'mae'],
                        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'max_features': ['auto', 'sqrt', 'log2', None],
                        'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        scorer= make_scorer(r2_score)
        forest_obj= RandomizedSearchCV(forest, param, cv=cv_sets, scoring=scorer, n_iter=100, random_state=4)
        forest_obj.fit(self.x_train, self.y_train)
        lst.append(self.regressionAccuracy(self.y_test, forest_obj.predict(self.x_test),'Random Forest',forest_obj))
        # AdaBoost
        ada=AdaBoostRegressor()
        cv_sets=ShuffleSplit(random_state=4)
        param= {'n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'loss': ['linear', 'square', 'exponential'],
                        'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        scorer= make_scorer(r2_score)
        ada_obj= RandomizedSearchCV(ada, param, cv=cv_sets, scoring=scorer, n_iter=100, random_state=4)
        ada_obj.fit(self.x_train, self.y_train)
        lst.append(self.regressionAccuracy(self.y_test, ada_obj.predict(self.x_test),'AdaBoost',ada_obj))
        # Xgbregress
        xgb=XGBRegressor()
        cv_sets=ShuffleSplit(random_state=4)
        param= {'n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'gamma': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'subsample': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'colsample_bytree': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'colsample_bylevel': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'reg_alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'reg_lambda': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        scorer= make_scorer(r2_score)
        xgb_obj= RandomizedSearchCV(xgb, param, cv=cv_sets, scoring=scorer, n_iter=100, random_state=4)
        xgb_obj.fit(self.x_train, self.y_train)
        lst.append(self.regressionAccuracy(self.y_test, xgb_obj.predict(self.x_test),'XGBoost',xgb_obj))
        # Gradient Boosting
        gb=GradientBoostingRegressor()
        cv_sets=ShuffleSplit(random_state=4)
        param= {'n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'max_features': ['auto', 'sqrt', 'log2', None],
                        'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        scorer= make_scorer(r2_score)
        gb_obj= RandomizedSearchCV(gb, param, cv=cv_sets, scoring=scorer, n_iter=100, random_state=4)
        gb_obj.fit(self.x_train, self.y_train)
        lst.append(self.regressionAccuracy(self.y_test, gb_obj.predict(self.x_test),'Gradient Boosting',gb_obj))
        # theil-sen
        theil=TheilSenRegressor()
        cv_sets=ShuffleSplit(random_state=4)
        param= {'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        scorer= make_scorer(r2_score)
        theil_obj= RandomizedSearchCV(theil, param, cv=cv_sets, scoring=scorer, n_iter=100, random_state=4)
        theil_obj.fit(self.x_train, self.y_train)
        lst.append(self.regressionAccuracy(self.y_test, theil_obj.predict(self.x_test),'Theil-Sen',theil_obj))
        # ransac regressor
        ransac=RANSACRegressor()
        cv_sets=ShuffleSplit(random_state=4)
        param= {'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'min_samples': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        'residual_threshold': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
        scorer= make_scorer(r2_score)
        ransac_obj= RandomizedSearchCV(ransac, param, cv=cv_sets, scoring=scorer, n_iter=100, random_state=4)
        ransac_obj.fit(self.x_train, self.y_train)
        lst.append(self.regressionAccuracy(self.y_test, ransac_obj.predict(self.x_test),'RANSAC',ransac_obj))
        # orthogonal matching pursuit
        omp=OrthogonalMatchingPursuit()
        cv_sets=ShuffleSplit(random_state=4)
        param= {'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'n_nonzero_coefs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
        scorer= make_scorer(r2_score)
        omp_obj= RandomizedSearchCV(omp, param, cv=cv_sets, scoring=scorer, n_iter=100, random_state=4)
        omp_obj.fit(self.x_train, self.y_train)
        lst.append(self.regressionAccuracy(self.y_test, omp_obj.predict(self.x_test),'Orthogonal Matching Pursuit',omp_obj))
        # lasso lars
        lasso_lars=LassoLars()
        cv_sets=ShuffleSplit(random_state=4)
        param= {'alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'max_iter': [None, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                        'normalize': [True, False],
                        'precompute': [True, False],
                        'fit_intercept': [True, False],
                        'positive': [True, False],
                        'max_n_alphas': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        'n_jobs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        scorer= make_scorer(r2_score)
        lasso_lars_obj= RandomizedSearchCV(lasso_lars, param, cv=cv_sets, scoring=scorer, n_iter=100, random_state=4)
        lasso_lars_obj.fit(self.x_train, self.y_train)
        lst.append(self.regressionAccuracy(self.y_test, lasso_lars_obj.predict(self.x_test),'Lasso Lars',lasso_lars_obj))
        # huber regression
        huber=HuberRegressor()
        cv_sets=ShuffleSplit(random_state=4)
        param= {'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'epsilon': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'max_iter': [None, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]}
        scorer= make_scorer(r2_score)
        huber_obj= RandomizedSearchCV(huber, param, cv=cv_sets, scoring=scorer, n_iter=100, random_state=4)
        huber_obj.fit(self.x_train, self.y_train)
        lst.append(self.regressionAccuracy(self.y_test, huber_obj.predict(self.x_test),'Huber Regression',huber_obj))
        # svr
        svr=SVR()
        cv_sets=ShuffleSplit(random_state=4)
        param= {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'epsilon': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                        'gamma': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'coef0': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'shrinking': [True, False],
                        'tol': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'cache_size': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'verbose': [True, False],
                        'max_iter': [None, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]}
        scorer= make_scorer(r2_score)
        svr_obj= RandomizedSearchCV(svr, param, cv=cv_sets, scoring=scorer, n_iter=100, random_state=4)
        svr_obj.fit(self.x_train, self.y_train)
        lst.append(self.regressionAccuracy(self.y_test, svr_obj.predict(self.x_test),'SVR',svr_obj))
        # passive aggressive
        passive_aggressive=PassiveAggressiveRegressor()
        cv_sets=ShuffleSplit(random_state=4)
        param= {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'loss': ['hinge', 'squared_hinge'],
                        'n_iter': [None, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                        'fit_intercept': [True, False],
                        'shuffle': [True, False],
                        'verbose': [True, False],
                        'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        scorer= make_scorer(r2_score)
        passive_obj= RandomizedSearchCV(passive_aggressive, param, cv=cv_sets, scoring=scorer, n_iter=100, random_state=4)
        passive_obj.fit(self.x_train, self.y_train)
        lst.append(self.regressionAccuracy(self.y_test, passive_obj.predict(self.x_test),'Passive Aggressive',passive_obj))
        # ardreg
        ardreg=ARDRegression()
        cv_sets=ShuffleSplit(random_state=4)
        param= {'n_iter': [None, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                        'normalize': [True, False],
                        'precompute': [True, False],
                        'fit_intercept': [True, False],
                        'positive': [True, False],
                        'max_iter': [None, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                        'n_jobs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        scorer= make_scorer(r2_score)
        ardreg_obj= RandomizedSearchCV(ardreg, param, cv=cv_sets, scoring=scorer, n_iter=100, random_state=4)
        ardreg_obj.fit(self.x_train, self.y_train)
        lst.append(self.regressionAccuracy(self.y_test, ardreg_obj.predict(self.x_test),'ARDRegression',ardreg_obj))
        # bayesian ridge
        bayesian_ridge=BayesianRidge()
        cv_sets=ShuffleSplit(random_state=4)
        param= {'n_iter': [None, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                        'tol': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'alpha_1': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'alpha_2': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'lambda_1': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'lambda_2': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'fit_intercept': [True, False],
                        'normalize': [True, False],
                        'copy_X': [True, False],
                        'n_jobs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        scorer= make_scorer(r2_score)
        bayesian_ridge_obj= RandomizedSearchCV(bayesian_ridge, param, cv=cv_sets, scoring=scorer, n_iter=100, random_state=4)
        bayesian_ridge_obj.fit(self.x_train, self.y_train)
        lst.append(self.regressionAccuracy(self.y_test, bayesian_ridge_obj.predict(self.x_test),'Bayesian Ridge',bayesian_ridge_obj))
        # bagging
        bagging=BaggingRegressor()
        cv_sets=ShuffleSplit(random_state=4)
        param= {'n_estimators': [None, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                        'max_samples': [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'max_features': [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'bootstrap': [True, False],
                        'bootstrap_features': [True, False],
                        'oob_score': [True, False],
                        'warm_start': [True, False],
                        'n_jobs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        scorer= make_scorer(r2_score)
        bagging_obj= RandomizedSearchCV(bagging, param, cv=cv_sets, scoring=scorer, n_iter=100, random_state=4)
        bagging_obj.fit(self.x_train, self.y_train)
        lst.append(self.regressionAccuracy(self.y_test, bagging_obj.predict(self.x_test),'Bagging',bagging_obj))
        # extra trees regressor
        extra_trees=ExtraTreesRegressor()
        cv_sets=ShuffleSplit(random_state=4)
        param= {'n_estimators': [None, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                        'max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'min_samples_split': [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'min_samples_leaf': [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'min_weight_fraction_leaf': [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'max_features': [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'max_leaf_nodes': [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        'min_impurity_decrease': [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'min_impurity_split': [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'bootstrap': [True, False],
                        'oob_score': [True, False],
                        'warm_start': [True, False],
                        'n_jobs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        scorer= make_scorer(r2_score)
        extra_trees_obj= RandomizedSearchCV(extra_trees, param, cv=cv_sets, scoring=scorer, n_iter=100, random_state=4)
        extra_trees_obj.fit(self.x_train, self.y_train)
        lst.append(self.regressionAccuracy(self.y_test, extra_trees_obj.predict(self.x_test),'Extra Trees',extra_trees_obj))

    def linearPrediction(self,x_predict):
        return self.lr_obj.predict(x_predict)
    
    def ridgePrediction(self,x_predict):
        return self.ridge_obj.predict(x_predict)
    
    def lassoPrediction(self,x_predict):
        return self.lasso_obj.predict(x_predict)
    
    def elasticNetPrediction(self,x_predict):
        return self.elastic_obj.predict(x_predict)
    
    def sgdrPrediction(self,x_predict):
        return self.sgd_obj.predict(x_predict)
    
    def knnPrediction(self,x_predict):
        return self.knn_obj.predict(x_predict)
    
    def decisionTreePredict(self,x_predict):
        return self.tree_obj.predict(x_predict)
    
    def randomForestPredict(self,x_predict):
        return self.forest_obj.predict(x_predict)
    
    def adaPrediction(self,x_predict):
        return self.ada_obj.predict(x_predict)
    
    def xgbregressorPrediction(self,x_predict):
        return self.xgb_obj.predict(x_predict)
    
    def gradientBoostPrediction(self,x_predict):
        return self.gb_obj.predict(x_predict)
    
    def theilsenPrediction(self,x_predict):
        return self.theil_obj.predict(x_predict)
    
    def ransacPrediction(self,x_predict):
        return self.ransac_obj.predict(x_predict)

    def orthogonalPrediction(self,x_predict):
        return self.omp_obj.predict(x_predict)
    
    def lassoLarsPrediction(self,x_predict):
        return self.lasso_lars_obj.predict(x_predict)
    
    def huberPrediction(self,x_predict):
        return self.huber_obj.predict(x_predict)
    
    def svrPrediction(self,x_predict):
        return self.svr_obj.predict(x_predict)
    
    def passivePrediction(self,x_predict):
        return self.passive_obj.predict(x_predict)
    
    def ardregPrediction(self,x_predict):
        return self.ardreg_obj.predict(x_predict)
    
    def bayesianRidgePrediction(self,x_predict):
        return self.bayesian_ridge_obj.predict(x_predict)
    
    def baggingPrediction(self,x_predict):
        return self.bagging_obj.predict(x_predict)
    
    def extraTreesPrediction(self,x_predict):
        return self.extra_trees_obj.predict(x_predict)
    






        