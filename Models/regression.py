# create Macine learning pipeline
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
import numpy as np
import pandas as pd

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



class regressionSingleModel:
    def __init__(self,X_train,y_train,X_test,y_test):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
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

# class for regression hyperparameter
class RegressionHyper:
    def __init__(self,x_train,y_train,x_test,y_test):
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
    
    def lasso(self):
        lasso=Lasso()
        cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
        parameters = {'alpha':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        'fit_intercept':[True, False],
                        'normalize':[True, False],
                        'precompute':[True, False],
                        'max_iter':[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                        'tol':[0.0001, 0.001, 0.01, 0.1, 1],
                        'selection':['cyclic', 'random'],
                        'copy_X':[True, False],
                        'warm_start':[True, False],
                        'positive':[True, False],
                        'random_state':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

        scorer = make_scorer(r2_score)
        lasso_obj = RandomizedSearchCV(lasso,
                            parameters,
                            scoring = scorer,
                            cv = cv_sets,
                            n_iter=15,
                            verbose=2,
                            n_jobs=-1,
                            random_state= 99)
        lasso_fit = lasso_obj.fit(self.x_train, self.y_train)
        lasso_opt = lasso_fit.best_estimator_
        self.model_lasso=lasso_opt
        self.model_lasso.fit(self.x_train, self.y_train)
        pred_lasso = self.model_lasso.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_lasso))
        print("MAE",mean_absolute_error(self.y_test,pred_lasso))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_lasso)))
        print("R2",r2_score(self.y_test, pred_lasso))

        return self.model_lasso
    
    def lasso_pred(self,test):
        pred_lasso = self.model_lasso.predict(test)
        return pred_lasso
    
    def ridge(self):
        ridge=Ridge()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'alpha':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        'fit_intercept':[True, False],
                        'normalize':[True, False],
                        'precompute':[True, False],
                        'max_iter':[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                        'tol':[0.0001, 0.001, 0.01, 0.1, 1],
                        'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                        'random_state':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

        scorer = make_scorer(r2_score)
        ridge_obj = RandomizedSearchCV(ridge,
                                    parameters,
                                    scoring = scorer,
                                    cv = cv_sets,
                                    n_iter=15,
                                    verbose=2,
                                    n_jobs=-1,
                                    random_state= 99)
        ridge_fit = ridge_obj.fit(self.x_train, self.y_train)
        ridge_opt = ridge_fit.best_estimator_
        self.model_ridge=ridge_opt
        self.model_ridge.fit(self.x_train, self.y_train)
        pred_ridge = self.model_ridge.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_ridge))
        print("MAE",mean_absolute_error(self.y_test,pred_ridge))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_ridge)))
        print("R2",r2_score(self.y_test, pred_ridge))
        return self.model_ridge

    def ridge_pred(self,test):
        pred_ridge = self.model_ridge.predict(test)
        return pred_ridge
    
    def elastic(self):
        el=ElasticNet()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'alpha':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        'l1_ratio':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        'fit_intercept':[True, False],
                        'normalize':[True, False],
                        'precompute':[True, False],
                        'max_iter':[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                        'tol':[0.0001, 0.001, 0.01, 0.1, 1],
                        'selection':['cyclic', 'random'],
                        'random_state':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

        scorer = make_scorer(r2_score)
        el_obj = RandomizedSearchCV(el,
                                    parameters,
                                    scoring = scorer,
                                    cv = cv_sets,
                                    n_iter=15,
                                    verbose=2,
                                    n_jobs=-1,
                                    random_state= 99)
        el_fit = el_obj.fit(self.x_train, self.y_train)
        el_opt = el_fit.best_estimator_
        self.model_el=el_opt
        self.model_el.fit(self.x_train, self.y_train)
        pred_el = self.model_el.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_el))
        print("MAE",mean_absolute_error(self.y_test,pred_el))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_el)))
        print("R2",r2_score(self.y_test, pred_el))
        return self.model_el
    
    def elastic_pred(self,test):
        pred_el = self.model_el.predict(test)
        return pred_el

    def svr(self):
        svr=SVR()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'C':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        'epsilon':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                        'degree':[1, 2, 3, 4, 5],
                        'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        'coef0':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        'shrinking':[True, False],
                        'tol':[0.0001, 0.001, 0.01, 0.1, 1],
                        'cache_size':[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                        'verbose':[True, False],
                        'max_iter':[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]}

        scorer = make_scorer(r2_score)
        svr_obj = RandomizedSearchCV(svr,
                                    parameters,
                                    scoring = scorer,
                                    cv = cv_sets,
                                    n_iter=15,
                                    verbose=2,
                                    n_jobs=-1,
                                    random_state= 99)
        svr_fit = svr_obj.fit(self.x_train, self.y_train)
        svr_opt = svr_fit.best_estimator_
        self.model_svr=svr_opt
        self.model_svr.fit(self.x_train, self.y_train)
        pred_svr = self.model_svr.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_svr))
        print("MAE",mean_absolute_error(self.y_test,pred_svr))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_svr)))
        print("R2",r2_score(self.y_test, pred_svr))
        return self.model_svr
    
    def svr_pred(self,test):
        pred_svr = self.model_svr.predict(test)
        return pred_svr

    def knn(self):
        knn=KNeighborsRegressor()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'n_neighbors':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'weights':['uniform', 'distance'],
                        'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
                        'leaf_size':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'p':[1, 2, 3],
                        'n_jobs':[-1]}

        scorer = make_scorer(r2_score)
        knn_obj = RandomizedSearchCV(knn,
                                    parameters,
                                    scoring = scorer,
                                    cv = cv_sets,
                                    n_iter=15,
                                    verbose=2,
                                    n_jobs=-1,
                                    random_state= 99)
        knn_fit = knn_obj.fit(self.x_train, self.y_train)
        knn_opt = knn_fit.best_estimator_
        self.model_knn=knn_opt
        self.model_knn.fit(self.x_train, self.y_train)
        pred_knn = self.model_knn.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_knn))
        print("MAE",mean_absolute_error(self.y_test,pred_knn))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_knn)))
        print("R2",r2_score(self.y_test, pred_knn))
        return self.model_knn
    
    def knn_pred(self,test):
        pred_knn = self.model_knn.predict(test)
        return pred_knn

    def rf(self):
        rf=RandomForestRegressor()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'n_estimators':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'criterion':['mse', 'mae'],
                        'max_features':['auto', 'sqrt', 'log2'],
                        'max_depth':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'min_samples_split':[2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'min_samples_leaf':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'bootstrap':[True, False],
                        'oob_score':[True, False],
                        'n_jobs':[-1]}

        scorer = make_scorer(r2_score)
        rf_obj = RandomizedSearchCV(rf,
                                    parameters,
                                    scoring = scorer,
                                    cv = cv_sets,
                                    n_iter=15,
                                    verbose=2,
                                    n_jobs=-1,
                                    random_state= 99)
        rf_fit = rf_obj.fit(self.x_train, self.y_train)
        rf_opt = rf_fit.best_estimator_
        self.model_rf=rf_opt
        self.model_rf.fit(self.x_train, self.y_train)
        pred_rf = self.model_rf.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_rf))
        print("MAE",mean_absolute_error(self.y_test,pred_rf))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_rf)))
        print("R2",r2_score(self.y_test, pred_rf))
        return self.model_rf
    
    def rf_pred(self,test):
        pred_rf = self.model_rf.predict(test)
        return pred_rf
    
    def gb(self):
        gb=GradientBoostingRegressor()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'loss':['ls', 'lad', 'huber', 'quantile'],
                        'learning_rate':[0.1, 0.05, 0.01],
                        'n_estimators':[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                        'max_depth':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'min_samples_split':[2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'min_samples_leaf':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'subsample':[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'max_features':['auto', 'sqrt', 'log2'],
                        'verbose':[True, False],
                        'n_jobs':[-1]}

        scorer = make_scorer(r2_score)
        gb_obj = RandomizedSearchCV(gb,
                                    parameters,
                                    scoring = scorer,
                                    cv = cv_sets,
                                    n_iter=15,
                                    verbose=2,
                                    n_jobs=-1,
                                    random_state= 99)
        gb_fit = gb_obj.fit(self.x_train, self.y_train)
        gb_opt = gb_fit.best_estimator_
        self.model_gb=gb_opt
        self.model_gb.fit(self.x_train, self.y_train)
        pred_gb = self.model_gb.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_gb))
        print("MAE",mean_absolute_error(self.y_test,pred_gb))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_gb)))
        print("R2",r2_score(self.y_test, pred_gb))
        return self.model_gb

    def gb_pred(self,test):
        pred_gb = self.model_gb.predict(test)
        return pred_gb

    def lr(self):
        lr=LinearRegression()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'fit_intercept':[True, False],
                'normalize':[True, False],
                'copy_X':[True, False],
                'n_jobs':[-1]}
        scorer = make_scorer(r2_score)
        lr_obj = RandomizedSearchCV(lr,
                    parameters,
                    scoring = scorer,
                    cv = cv_sets,
                    n_iter=15,
                    verbose=2,
                    n_jobs=-1,
                    random_state= 99)
        lr_fit = lr_obj.fit(self.x_train, self.y_train)
        lr_opt = lr_fit.best_estimator_
        self.model_lr=lr_opt
        self.model_lr.fit(self.x_train, self.y_train)
        pred_lr = self.model_lr.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_lr))
        print("MAE",mean_absolute_error(self.y_test,pred_lr))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_lr)))
        print("R2",r2_score(self.y_test, pred_lr))
        return self.model_lr
    
    def lr_pred(self,test):
        pred_lr = self.model_lr.predict(test)
        return pred_lr
    
    def xgboost(self):
        xg=XGBRegressor()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'learning_rate':[0.1, 0.05, 0.010,0.005, 0.001,0.19,0.15,0.11,0.09,0.07,0.05,0.03,0.01],
                        'n_estimators':[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                        'max_depth':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'min_samples_split':[2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'min_samples_leaf':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'subsample':[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'max_features':['auto', 'sqrt', 'log2'],
                        'verbose':[True, False],
                        'n_jobs':[-1]}

        scorer = make_scorer(r2_score)
        xg_obj = RandomizedSearchCV(xg,
                                    parameters,
                                    scoring = scorer,
                                    cv = cv_sets,
                                    n_iter=15,
                                    verbose=2,
                                    n_jobs=-1,
                                    random_state= 99)
        xg_fit = xg_obj.fit(self.x_train, self.y_train)
        xg_opt = xg_fit.best_estimator_
        self.model_xg=xg_opt
        self.model_xg.fit(self.x_train, self.y_train)
        pred_xg = self.model_xg.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_xg))
        print("MAE",mean_absolute_error(self.y_test,pred_xg))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_xg)))
        print("R2",r2_score(self.y_test, pred_xg))
        return self.model_xg
    
    def xg_pred(self,test):
        pred_xg = self.model_xg.predict(test)
        return pred_xg

    def sgdregressor(self):
        sgd=SGDRegressor()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'loss':['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                        'penalty':['none', 'l2', 'l1', 'elasticnet'],
                        'alpha':[0.0001, 0.001, 0.01, 0.1, 1],
                        'learning_rate':['constant', 'optimal', 'invscaling', 'adaptive'],
                        'eta0':[0.0001, 0.001, 0.01, 0.1, 1],
                        'power_t':[0.5, 0.75, 1],
                        'n_jobs':[-1]}

        scorer = make_scorer(r2_score)
        sgd_obj = RandomizedSearchCV(sgd,
                                    parameters,
                                    scoring = scorer,
                                    cv = cv_sets,
                                    n_iter=15,
                                    verbose=2,
                                    n_jobs=-1,
                                    random_state= 99)
        sgd_fit = sgd_obj.fit(self.x_train, self.y_train)
        sgd_opt = sgd_fit.best_estimator_
        self.model_sgd=sgd_opt
        self.model_sgd.fit(self.x_train, self.y_train)
        pred_sgd = self.model_sgd.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_sgd))
        print("MAE",mean_absolute_error(self.y_test,pred_sgd))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_sgd)))
        print("R2",r2_score(self.y_test, pred_sgd))
        return self.model_sgd
    
    def sgd_pred(self,test):
        pred_sgd = self.model_sgd.predict(test)
        return pred_sgd
    
    def decisiontree(self):
        dt=DecisionTreeRegressor()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'criterion':['mse', 'friedman_mse', 'mae'],
                        'splitter':['best', 'random'],
                        'max_depth':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'min_samples_split':[2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'min_samples_leaf':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'max_features':['auto', 'sqrt', 'log2'],
                        'random_state':[99]}
        scorer = make_scorer(r2_score)
        dt_obj = RandomizedSearchCV(dt,
                    parameters,
                    scoring = scorer,
                    cv = cv_sets,
                    n_iter=15,
                    verbose=2,
                    n_jobs=-1,
                    random_state= 99)
        dt_fit = dt_obj.fit(self.x_train, self.y_train)
        dt_opt = dt_fit.best_estimator_
        self.model_dt=dt_opt
        self.model_dt.fit(self.x_train, self.y_train)
        pred_dt = self.model_dt.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_dt))
        print("MAE",mean_absolute_error(self.y_test,pred_dt))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_dt)))
        print("R2",r2_score(self.y_test, pred_dt))
        return self.model_dt
    
    def dt_pred(self,test):
        pred_dt = self.model_dt.predict(test)
        return pred_dt

    def ada_boost(self):
        ada=AdaBoostRegressor()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'n_estimators':[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
                        'learning_rate':[0.01, 0.1, 1, 10, 100],
                        'loss':['linear', 'square', 'exponential'],
                        'random_state':[99],
                        'n_jobs':[-1] }
        scorer = make_scorer(r2_score)
        ada_obj = RandomizedSearchCV(ada,
                    parameters,
                    scoring = scorer,
                    cv = cv_sets,
                    n_iter=15,
                    verbose=2,
                    n_jobs=-1,
                    random_state= 99)
        ada_fit = ada_obj.fit(self.x_train, self.y_train)
        ada_opt = ada_fit.best_estimator_
        self.model_ada=ada_opt
        self.model_ada.fit(self.x_train, self.y_train)
        pred_ada = self.model_ada.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_ada))
        print("MAE",mean_absolute_error(self.y_test,pred_ada))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_ada)))
        print("R2",r2_score(self.y_test, pred_ada))
        return self.model_ada
    
    def ada_pred(self,test):
        pred_ada = self.model_ada.predict(test)
        return pred_ada

    def theilsenregressor(self):
        theil=TheilSenRegressor()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'random_state':[1,10,32,35,40,42,45,99],
                'n_jobs':[-1]}
        scorer = make_scorer(r2_score)
        theil_obj = RandomizedSearchCV(theil,
            parameters,
            scoring = scorer,
            cv = cv_sets,
            n_iter=15,
            verbose=2,
            n_jobs=-1,
            random_state= 99)
        theil_fit = theil_obj.fit(self.x_train, self.y_train)
        theil_opt = theil_fit.best_estimator_
        self.model_theil=theil_opt
        self.model_theil.fit(self.x_train, self.y_train)
        pred_theil = self.model_theil.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_theil))
        print("MAE",mean_absolute_error(self.y_test,pred_theil))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_theil)))
        print("R2",r2_score(self.y_test, pred_theil))
        return self.model_theil
    
    def theil_pred(self,test):
        pred_theil = self.model_theil.predict(test)
        return pred_theil

    def ransacregressor(self):
        ransac=RANSACRegressor()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'random_state':[1,10,32,35,40,42,45,99],
                'n_jobs':[-1]}
        scorer = make_scorer(r2_score)
        ransac_obj = RandomizedSearchCV(ransac,
            parameters,
            scoring = scorer,
            cv = cv_sets,
            n_iter=15,
            verbose=2,
            n_jobs=-1,
            random_state= 99)
        ransac_fit = ransac_obj.fit(self.x_train, self.y_train)
        ransac_opt = ransac_fit.best_estimator_
        self.model_ransac=ransac_opt
        self.model_ransac.fit(self.x_train, self.y_train)
        pred_ransac = self.model_ransac.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_ransac))
        print("MAE",mean_absolute_error(self.y_test,pred_ransac))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_ransac)))
        print("R2",r2_score(self.y_test, pred_ransac))
        return self.model_ransac
    
    def ransac_pred(self,test):
        pred_ransac = self.model_ransac.predict(test)
        return pred_ransac
    
    def orthogonalmatchingpursuit(self):
        omp=OrthogonalMatchingPursuit()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'random_state':[1,10,32,35,40,42,45,99],
                'n_jobs':[-1]}
        scorer = make_scorer(r2_score)
        omp_obj = RandomizedSearchCV(omp,
                                    parameters,
                                    scoring = scorer,
                                    cv = cv_sets,
                                    n_iter=15,
                                    verbose=2,
                                    n_jobs=-1,
                                    random_state= 99)
        omp_fit = omp_obj.fit(self.x_train, self.y_train)
        omp_opt = omp_fit.best_estimator_
        self.model_omp=omp_opt
        self.model_omp.fit(self.x_train, self.y_train)
        pred_omp = self.model_omp.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_omp))
        print("MAE",mean_absolute_error(self.y_test,pred_omp))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_omp)))
        print("R2",r2_score(self.y_test, pred_omp))
        return self.model_omp
    
    def ortho_pred(self,test):
        pred_omp = self.model_omp.predict(test)
        return pred_omp

    def lassolars(self):
        lasso=Lasso()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'random_state':[1,10,32,35,40,42,45,99],
                'n_jobs':[-1]}
        scorer = make_scorer(r2_score)
        lasso_obj = RandomizedSearchCV(lasso,
                                        parameters,
                                        scoring = scorer,
                                        cv = cv_sets,
                                        n_iter=15,
                                        verbose=2,
                                        n_jobs=-1,
                                        random_state= 99)
        lasso_fit = lasso_obj.fit(self.x_train, self.y_train)
        lasso_opt = lasso_fit.best_estimator_
        self.model_lasso=lasso_opt
        self.model_lasso.fit(self.x_train, self.y_train)
        pred_lasso = self.model_lasso.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_lasso))
        print("MAE",mean_absolute_error(self.y_test,pred_lasso))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_lasso)))
        print("R2",r2_score(self.y_test, pred_lasso))
        return self.model_lasso

    def lasso_pred(self,test):
        pred_lasso = self.model_lasso.predict(test)
        return pred_lasso

    def lars(self):
        lars=Lars()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'random_state':[1,10,32,35,40,42,45,99],
                'n_jobs':[-1]}
        scorer = make_scorer(r2_score)
        lars_obj = RandomizedSearchCV(lars,
                                        parameters,
                                        scoring = scorer,
                                        cv = cv_sets,
                                        n_iter=15,
                                        verbose=2,
                                        n_jobs=-1,
                                        random_state= 99)
        lars_fit = lars_obj.fit(self.x_train, self.y_train)
        lars_opt = lars_fit.best_estimator_
        self.model_lars=lars_opt
        self.model_lars.fit(self.x_train, self.y_train)
        pred_lars = self.model_lars.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_lars))
        print("MAE",mean_absolute_error(self.y_test,pred_lars))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_lars)))
        print("R2",r2_score(self.y_test, pred_lars))
        return self.model_lars
    
    def lars_pred(self,test):
        pred_lars = self.model_lars.predict(test)
        return pred_lars

    def huberregressor(self):
        huber=HuberRegressor()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'random_state':[1,10,32,35,40,42,45,99],
            'n_jobs':[-1]}
        scorer = make_scorer(r2_score)
        huber_obj = RandomizedSearchCV(huber,
                    parameters,
                    scoring = scorer,
                    cv = cv_sets,
                    n_iter=15,
                    verbose=2,
                    n_jobs=-1,
                    random_state= 99)
        huber_fit = huber_obj.fit(self.x_train, self.y_train)
        huber_opt = huber_fit.best_estimator_
        self.model_huber=huber_opt
        self.model_huber.fit(self.x_train, self.y_train)
        pred_huber = self.model_huber.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_huber))
        print("MAE",mean_absolute_error(self.y_test,pred_huber))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_huber)))
        print("R2",r2_score(self.y_test, pred_huber))
        return self.model_huber
    
    def huber_pred(self,test):
        pred_huber = self.model_huber.predict(test)
        return pred_huber

    def passiveaggressiveregressor(self):
        passiveaggressive=PassiveAggressiveRegressor()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'random_state':[1,10,32,35,40,42,45,99],
            'n_jobs':[-1]}
        scorer = make_scorer(r2_score)
        passiveaggressive_obj = RandomizedSearchCV(passiveaggressive,
                    parameters,
                    scoring = scorer,
                    cv = cv_sets,
                    n_iter=15,
                    verbose=2,
                    n_jobs=-1,
                    random_state= 99)
        passiveaggressive_fit = passiveaggressive_obj.fit(self.x_train, self.y_train)
        passiveaggressive_opt = passiveaggressive_fit.best_estimator_
        self.model_passiveaggressive=passiveaggressive_opt
        self.model_passiveaggressive.fit(self.x_train, self.y_train)
        pred_passiveaggressive = self.model_passiveaggressive.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_passiveaggressive))
        print("MAE",mean_absolute_error(self.y_test,pred_passiveaggressive))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_passiveaggressive)))
        print("R2",r2_score(self.y_test, pred_passiveaggressive))
        return self.model_passiveaggressive
    
    def passiveaggressive_pred(self,test):
        pred_passiveaggressive = self.model_passiveaggressive.predict(test)
        return pred_passiveaggressive

    def ardregression(self):
        ard=ARDRegression()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'random_state':[1,10,32,35,40,42,45,99],
            'n_jobs':[-1]}
        scorer = make_scorer(r2_score)
        ard_obj = RandomizedSearchCV(ard,
                    parameters,
                    scoring = scorer,
                    cv = cv_sets,
                    n_iter=15,
                    verbose=2,
                    n_jobs=-1,
                    random_state= 99)
        ard_fit = ard_obj.fit(self.x_train, self.y_train)
        ard_opt = ard_fit.best_estimator_
        self.model_ard=ard_opt
        self.model_ard.fit(self.x_train, self.y_train)
        pred_ard = self.model_ard.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_ard))
        print("MAE",mean_absolute_error(self.y_test,pred_ard))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_ard)))
        print("R2",r2_score(self.y_test, pred_ard))
        return self.model_ard

    def ard_pred(self,test):
        pred_ard = self.model_ard.predict(test)
        return pred_ard
    
    def bayesianridge(self):
        bayesianridge=BayesianRidge()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'random_state':[1,10,32,35,40,42,45,99],
            'n_jobs':[-1]}
        scorer = make_scorer(r2_score)
        bayesianridge_obj = RandomizedSearchCV(bayesianridge,
                    parameters,
                    scoring = scorer,
                    cv = cv_sets,
                    n_iter=15,
                    verbose=2,
                    n_jobs=-1,
                    random_state= 99)
        bayesianridge_fit = bayesianridge_obj.fit(self.x_train, self.y_train)
        bayesianridge_opt = bayesianridge_fit.best_estimator_
        self.model_bayesianridge=bayesianridge_opt
        self.model_bayesianridge.fit(self.x_train, self.y_train)
        pred_bayesianridge = self.model_bayesianridge.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_bayesianridge))
        print("MAE",mean_absolute_error(self.y_test,pred_bayesianridge))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_bayesianridge)))
        print("R2",r2_score(self.y_test, pred_bayesianridge))
        return self.model_bayesianridge
    
    def bayesianridge_pred(self,test):
        pred_bayesianridge = self.model_bayesianridge.predict(test)
        return pred_bayesianridge

    def baggingregressor(self):
        bagging = BaggingRegressor()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'random_state':[1,10,32,35,40,42,45,99],
            'n_jobs':[-1]}
        scorer = make_scorer(r2_score)
        bagging_obj = RandomizedSearchCV(bagging,
                    parameters,
                    scoring = scorer,
                    cv = cv_sets,
                    n_iter=15,
                    verbose=2,
                    n_jobs=-1,
                    random_state= 99)
        bagging_fit = bagging_obj.fit(self.x_train, self.y_train)
        bagging_opt = bagging_fit.best_estimator_
        self.model_bagging=bagging_opt
        self.model_bagging.fit(self.x_train, self.y_train)
        pred_bagging = self.model_bagging.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_bagging))
        print("MAE",mean_absolute_error(self.y_test,pred_bagging))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_bagging)))
        print("R2",r2_score(self.y_test, pred_bagging))
        return self.model_bagging

    def bagging_pred(self,test):
        pred_bagging = self.model_bagging.predict(test)
        return pred_bagging
    
    def extratreesregressor(self):
        extratrees = ExtraTreesRegressor()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'random_state':[1,10,32,35,40,42,45,99],
            'n_jobs':[-1]}
        scorer = make_scorer(r2_score)
        extratrees_obj = RandomizedSearchCV(extratrees,
            parameters,
            scoring = scorer,
            cv = cv_sets,
            n_iter=15,
            verbose=2,
            n_jobs=-1,
            random_state= 99)
        
        extratrees_fit = extratrees_obj.fit(self.x_train, self.y_train)
        extratrees_opt = extratrees_fit.best_estimator_
        self.model_extratrees=extratrees_opt
        self.model_extratrees.fit(self.x_train, self.y_train)
        pred_extratrees = self.model_extratrees.predict(self.x_test)
        print("MSE",mean_squared_error(self.y_test,pred_extratrees))
        print("MAE",mean_absolute_error(self.y_test,pred_extratrees))
        print("RMSE",math.sqrt(mean_squared_error(self.y_test,pred_extratrees)))
        print("R2",r2_score(self.y_test, pred_extratrees))
        return self.model_extratrees
    
    def extratrees_pred(self,test):
        pred_extratrees = self.model_extratrees.predict(test)
        return pred_extratrees






    


    







                    
        


