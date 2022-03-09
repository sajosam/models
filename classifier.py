# import requered libraries for classifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LogisticRegression,RidgeClassifier,SGDClassifier,Lasso
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV

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

class classifierModel:
    def __init__(self,x,y,test_size=0.2,random_state=0):
        self.x = x
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        # self.pred = pred
        self.model=[LogisticRegression(),KNeighborsClassifier(),GaussianNB(),
                    MultinomialNB(),BaggingClassifier(),ExtraTreesClassifier(),
                    RidgeClassifier(),SGDClassifier(),RandomForestClassifier(),
                    XGBClassifier(),AdaBoostClassifier(),BernoulliNB(),GradientBoostingClassifier(),DecisionTreeClassifier(),SVC(),Lasso()]
        self.model_name=['Logistic Regression','KNeighborsClassifier','GaussianNB','MultinomialNB',
                'BaggingClassifier','ExtraTreesClassifier','RidgeClassifier','SGDClassifier',
                'RandomForestClassifier','XGBClassifier','AdaBoostClassifier',
                'BernoulliNB','GradientBoostingClassifier','DecisionTreeClassifier','SVC','Lasso']
        self.model_table=[]
        self.mod=[]
    
    def cl_accuracy(self,y_test_f,y_pred_f,model_name,model_obj):
        acc=accuracy_score(y_test_f, y_pred_f)
        confusion=confusion_matrix(y_test_f, y_pred_f)
        roc=roc_auc_score(y_test_f, y_pred_f)
        f1=f1_score(y_test_f, y_pred_f)
        recall=recall_score(y_test_f, y_pred_f)
        precision=precision_score(y_test_f, y_pred_f)
        return {'model name':model_name,'accuracy':acc,'confusion':confusion,'roc':roc,'f1':f1,'recall':recall,'precision':precision,'model object':model_obj}
    def Model(self):
        # split data into train and test
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x,self.y, test_size=self.test_size, random_state=self.random_state)
        # scaling
        self.x_train=Scaling(self.x_train).standard_scaler()
        self.x_test=Scaling(self.x_test).standard_scaler()

        # x_train, x_test, pred = Scaling(x_train,x_test,pred).standard_scaler()

        # model
        lst=[]
        for m, m_n in zip(self.model,self.model_name):
            model=m.fit(self.x_train, self.y_train)
            y_pred=model.predict(self.x_test)
            lst.append(self.cl_accuracy(self.y_test, y_pred,m_n,model))
        self.model_table=pd.DataFrame(lst)
        return self.model_table
    
    def hyperParameter(self):
        lst=[]
        #logistic regression
        l=LogisticRegression(random_state=42)
        cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
        parameters = {'C':[0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
                'penalty':['l1', 'l2'],
                'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

        scorer = make_scorer(f1_score)
        l_obj = RandomizedSearchCV(l,
                                    parameters,
                                    scoring = scorer,
                                    cv = cv_sets,
                                    verbose=2,
                                    n_jobs=-1,
                                    n_iter=15,
                                    random_state= 99)
        l_fit = l_obj.fit(self.x_train, self.y_train)
        l_opt = l_fit.best_estimator_
        self.model_l=l_opt
        self.model_l.fit(self.x_train, self.y_train)
        pred_l = self.model_l.predict(self.x_test)
        lst.append(self.cl_accuracy(self.y_test, pred_l,'logistic_regression',self.model_l))


        #knn
        knn=KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
        parameters = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10],
            'metric':['minkowski', 'euclidean', 'manhattan'],
            'weights':['uniform', 'distance'],
            'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size':[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'metric_params':[{}, {'p': 2}, {'p': 1}, {'p': 3}],
            'p':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        scorer = make_scorer(f1_score)
        knn_obj = RandomizedSearchCV(knn,
            parameters,
            scoring = scorer,
            cv = cv_sets,
            verbose=2,
            n_jobs=-1,
            n_iter=15,
            random_state= 99)
        knn_fit = knn_obj.fit(self.x_train, self.y_train)
        knn_opt = knn_fit.best_estimator_
        self.model_knn=knn_opt
        self.model_knn.fit(self.x_train, self.y_train)
        pred_knn = self.model_knn.predict(self.x_test)
        lst.append(self.cl_accuracy(self.y_test, pred_knn,'KNeighborsClassifier' ,self.model_knn))

        #ada boost
        ada_classifier = AdaBoostClassifier(random_state=42)
        cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
        parameters = {'n_estimators':[500, 1000, 1500, 2000], 
                     'learning_rate':[0.05, 0.1, 0.15, 0.2],
                     'algorithm':['SAMME', 'SAMME.R'],
                     'random_state':[None,1,2,10,9,32,35,42,99]}

        scorer = make_scorer(f1_score)
        ada_obj = RandomizedSearchCV(ada_classifier, 
                                     parameters, 
                                     scoring = scorer, 
                                     cv = cv_sets,
                                     verbose=2,
                                     n_iter=15,
                                     random_state= 99)
        ada_fit = ada_obj.fit(self.x_train, self.y_train)
        ada_opt = ada_fit.best_estimator_
        self.model_ada=ada_opt
        self.model_ada.fit(self.x_train, self.y_train)
        pred_ada = self.model_ada.predict(self.x_test)
        lst.append(self.cl_accuracy(self.y_test, pred_ada,'AdaBoostClassifier' ,self.model_ada))

        #random forest

        rf=RandomForestClassifier(random_state=42)
        cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
        parameters = {'n_estimators':[100,200,500,1000, 1500, 2000],
                    'criterion':['gini', 'entropy'],
                    'min_weight_fraction_leaf':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    'max_leaf_nodes':[None, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                    'min_impurity_decrease':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    'min_impurity_split':[None, 1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    'bootstrap':[True, False],
                    'oob_score':[True, False],
                    'warm_start':[True, False],
                    'class_weight':['balanced', None],
                    'ccp_alpha':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    'random_state':[None,1,2,10,9,32,35,42,99],
                    'learning_rate':[0.05, 0.1, 0.15, 0.2], 
                    'max_depth':[3,5,7,9,10,20,30,40,50,None],
                    'min_samples_split':[2,3,4,5,8,10],
                    'min_samples_leaf':[1,2,3,4,5],
                    'max_features':['auto', 'sqrt', 'log2'],
                    'max_terminal_nodes':[None, 2, 3, 4, 5, 6, 7, 8, 9, 10],    
                    'max_samples':[None, 0.5, 0.75, 1.0],
                    'bootstrap':[True, False]}
        scorer = make_scorer(f1_score)
        rf_obj = RandomizedSearchCV(estimator=rf,
                    param_distributions=parameters,
                    n_iter=15,
                    scoring = scorer,
                    cv = cv_sets,
                    verbose=2,
                    n_jobs=-1,
                    random_state= 99)
        rf_fit = rf_obj.fit(self.x_train, self.y_train)
        rf_opt = rf_fit.best_estimator_
        self.model_rf=rf_opt
        self.model_rf.fit(self.x_train, self.y_train)
        pred_rf = self.model_rf.predict(self.x_test)
        lst.append(self.cl_accuracy(self.y_test, pred_rf,'RandomForestClassifier' ,self.model_rf))

        #gradient boosting
        gb=GradientBoostingClassifier(random_state=42)
        cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
        parameters = {'n_estimators':[500, 1000, 1500, 2000],
                        'loss':['deviance', 'exponential'],
                        'criterion':['friedman_mse', 'mse', 'mae'],
                        'min_impurity_decrease':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        # 'min_impurity_split':[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        # 'init':['zero', 'uniform', 'random'],
                        'random_state':[None,1,9,10,11,32,35,42,99],
                        'warm_start':[True, False],
                        'validation_fraction':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'n_iter_no_change':[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'ccp_alpha':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'learning_rate':[0.05, 0.1, 0.15, 0.2],
                        'max_depth':[3,5,7,9],
                        'min_samples_split':[2,3,4,5],
                        'min_samples_leaf':[1,2,3,4],  
                        'min_weight_fraction_leaf':[0.0, 0.1, 0.2, 0.3],
                        'max_leaf_nodes':[None, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                        'subsample':[0.5, 0.75,0.8,0.9, 1.0],
                        'max_features':['auto', 'sqrt', 'log2']}
        scorer = make_scorer(f1_score)
        gb_obj = RandomizedSearchCV(gb,
                                    parameters,
                                    scoring = scorer,
                                    cv = cv_sets,   
                                    verbose=2,
                                    n_iter=15,
                                    random_state= 99)
        gb_fit = gb_obj.fit(self.x_train, self.y_train)
        gb_opt = gb_fit.best_estimator_
        self.model_gb=gb_opt
        self.model_gb.fit(self.x_train, self.y_train)
        pred_gb = self.model_gb.predict(self.x_test)
        lst.append(self.cl_accuracy(self.y_test, pred_gb,'GradientBoostingClassifier' ,self.model_gb))

        #bernoullinb
        nb=BernoulliNB()
        cv_sets = ShuffleSplit(random_state = 4)
        parameters = {'alpha':[0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
            'binarize':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'fit_prior':[True, False]}

        scorer = make_scorer(f1_score)
        nb_obj = RandomizedSearchCV(nb,
            parameters,
            scoring = scorer,
            cv = cv_sets,
            verbose=2,
            n_jobs=-1,
            n_iter=15,
            random_state= 99)
        nb_fit = nb_obj.fit(self.x_train, self.y_train)
        nb_opt = nb_fit.best_estimator_
        self.model_nb=nb_opt
        self.model_nb.fit(self.x_train, self.y_train)
        pred_nb = self.model_nb.predict(self.x_test)
        lst.append(self.cl_accuracy(self.y_test, pred_nb,'BernoulliNB' ,self.model_nb))

        #multinomialnb
        mb=MultinomialNB()
        cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
        parameters = {'alpha':[0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
            'fit_prior':[True, False],
            'class_prior':[None, [0.1, 0.9]]}
        scorer = make_scorer(f1_score)
        mb_obj = RandomizedSearchCV(mb,
            parameters,
            scoring = scorer,
            cv = cv_sets,
            verbose=2,
            n_iter=15,
            n_jobs=-1,
            random_state= 99)
        mb_fit = mb_obj.fit(self.x_train, self.y_train)
        mb_opt = mb_fit.best_estimator_
        self.model_mb=mb_opt
        self.model_mb.fit(self.x_train, self.y_train)
        pred_mb = self.model_mb.predict(self.x_test)
        lst.append(self.cl_accuracy(self.y_test, pred_mb,'MultinomialNB' ,self.model_mb))

        #gaussiannb
        gnb=GaussianNB()
        cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
        parameters = {'var_smoothing':[1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
        'priors':[None, [0.1, 0.9]]}
        scorer = make_scorer(f1_score)
        gnb_obj = RandomizedSearchCV(gnb,
                                    parameters,
                                    scoring = scorer,
                                    cv = cv_sets,
                                    verbose=2,
                                    n_iter=15,
                                    n_jobs=-1,
                                    random_state= 99)
        gnb_fit = gnb_obj.fit(self.x_train, self.y_train)
        gnb_opt = gnb_fit.best_estimator_
        self.model_gnb=gnb_opt
        self.model_gnb.fit(self.x_train, self.y_train)
        pred_gnb = self.model_gnb.predict(self.x_test)
        lst.append(self.cl_accuracy(self.y_test, pred_gnb,'GaussianNB' ,self.model_gnb))
        #rxgboost
        xgb=XGBClassifier()
        cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
        parameters = {'learning_rate':[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                                        'booster':['gbtree', 'gblinear', 'dart'],
                                        'n_estimators':[50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                                        'max_leaf_nodes':[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                        'max_delta_step':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                        'subsample':[0.5, 0.6, 0.7, 0.8, 0.9, 1],
                                        'colsample_bytree':[0.5, 0.6, 0.7, 0.8, 0.9, 1],
                                        'colsample_bylevel':[0.5, 0.6, 0.7, 0.8, 0.9, 1],
                                        'lambda':[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                                        'alpha':[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                                        'scale_pos_weight':[1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                        'eval_metric':['mae', 'auc', 'logloss'],
                                        'seed':[99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80,
                                        79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57,
                                        56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25,
                                        24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                                        'max_depth':[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
                                        'min_child_weight':[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
                                        'gamma':[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                                        'subsample':[0.6, 0.7, 0.8, 0.9, 1],
                                        'colsample_bytree':[0.6, 0.7, 0.8, 0.9, 1],
                                        'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
                                        'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]}
        scorer = make_scorer(f1_score)
        xgb_obj = RandomizedSearchCV(xgb,
                                        parameters,
                                        scoring = scorer,
                                        cv = cv_sets,
                                        verbose=2,
                                        n_iter=15,
                                        n_jobs=-1,
                                        random_state= 99)
        xgb_fit = xgb_obj.fit(self.x_train, self.y_train)
        xgb_opt = xgb_fit.best_estimator_
        self.model_xgb=xgb_opt
        self.model_xgb.fit(self.x_train, self.y_train)
        pred_xgb = self.model_xgb.predict(self.x_test)
        lst.append(self.cl_accuracy(self.y_test, pred_xgb,'XGBoost', self.model_xgb))

        #bagging
        bg=BaggingClassifier()
        cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
        parameters = {'n_estimators':[50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                        'max_depth':[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
                        'min_child_weight':[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
                        'gamma':[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        'subsample':[0.6, 0.7, 0.8, 0.9, 1],
                        'colsample_bytree':[0.6, 0.7, 0.8, 0.9, 1],
                        'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
                        'bootstrap':[True, False],
                        'oob_score':[True, False],
                        'bootstrap_features':[True, False],
                        'warm_start':[True, False],
                        'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]}
        scorer = make_scorer(f1_score)
        bg_obj = RandomizedSearchCV(bg,
                                    parameters,
                                    scoring = scorer,
                                    cv = cv_sets,
                                    verbose=2,
                                    n_iter=15,
                                    n_jobs=-1,
                                    random_state= 99)

        bg_fit = bg_obj.fit(self.x_train, self.y_train)
        bg_opt = bg_fit.best_estimator_
        self.model_bg=bg_opt
        self.model_bg.fit(self.x_train, self.y_train)
        pred_bg = self.model_bg.predict(self.x_test)
        lst.append(self.cl_accuracy(self.y_test, pred_bg,'Bagging' ,self.model_bg))

        #extra trees
        xt=ExtraTreesClassifier()
        cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
        parameters = {'n_estimators':[50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                'max_depth':[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
                'min_child_weight':[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
                'gamma':[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                'subsample':[0.6, 0.7, 0.8, 0.9, 1],
                'colsample_bytree':[0.6, 0.7, 0.8, 0.9, 1],
                'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
                'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]}
        scorer = make_scorer(f1_score)
        xt_obj = RandomizedSearchCV(xt,
                    parameters,
                    scoring = scorer,
                    cv = cv_sets,
                    verbose=2,
                    n_iter=15,
                    n_jobs=-1,
                    random_state= 99)
        
        xt_fit = xt_obj.fit(self.x_train, self.y_train)
        xt_opt = xt_fit.best_estimator_
        self.model_xt=xt_opt
        self.model_xt.fit(self.x_train, self.y_train)
        pred_xt = self.model_xt.predict(self.x_test)
        lst.append(self.cl_accuracy(self.y_test, pred_xt,'ExtraTrees' ,self.model_xt))

        #ridge
        rc=RidgeClassifier()
        cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
        parameters = {'alpha':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
        scorer = make_scorer(f1_score)
        rc_obj = RandomizedSearchCV(rc,
                            parameters,
                            scoring = scorer,
                            cv = cv_sets,
                            n_iter=15,
                            verbose=2,
                            n_jobs=-1,
                            random_state= 99)
        rc_fit = rc_obj.fit(self.x_train, self.y_train)
        rc_opt = rc_fit.best_estimator_
        self.model_rc=rc_opt
        self.model_rc.fit(self.x_train, self.y_train)
        pred_rc = self.model_rc.predict(self.x_test)
        lst.append(self.cl_accuracy(self.y_test, pred_rc,'Ridge' ,self.model_rc))
    
        #lasso
        lasso=Lasso()
        cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
        parameters = {'alpha':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
        scorer = make_scorer(f1_score)
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
        lst.append(self.cl_accuracy(self.y_test, pred_lasso,'Lasso' ,self.model_lasso))

        #sgdc
        sgdc=SGDClassifier()    
        cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
        parameters = {'loss':['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                    'penalty':['l1', 'l2', 'elasticnet'],
                    'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
                    'l1_ratio':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
                    'fit_intercept':[True, False],
                    'max_iter':[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                    'tol':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}
        scorer = make_scorer(f1_score)
        sgdc_obj = RandomizedSearchCV(sgdc,
                            parameters,
                            scoring = scorer,
                            cv = cv_sets,
                            n_iter=15,
                            verbose=2,
                            n_jobs=-1,
                            random_state= 99)
        sgdc_fit = sgdc_obj.fit(self.x_train, self.y_train)
        sgdc_opt = sgdc_fit.best_estimator_
        self.model_sgdc=sgdc_opt
        self.model_sgdc.fit(self.x_train, self.y_train)
        pred_sgdc = self.model_sgdc.predict(self.x_test)
        lst.append(self.cl_accuracy(self.y_test, pred_sgdc,'SGDC' ,self.model_sgdc))

        #decision tree
        dt=DecisionTreeClassifier()
        cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
        parameters = {'criterion':['gini', 'entropy'],
                    'splitter':['best', 'random'],  # 'best' is the default
                    'max_depth':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                    'min_samples_split':[2, 5, 10],
                    'min_samples_leaf':[1, 2, 4, 6, 8, 10],
                    'min_weight_fraction_leaf':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    'max_features':['auto', 'sqrt', 'log2'],
                    'random_state':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
        scorer = make_scorer(f1_score)
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
        lst.append(self.cl_accuracy(self.y_test, pred_dt,'Decision Tree' ,self.model_dt))

        #svc
        svc=SVC()
        cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
        parameters = {'C':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                    'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                    'degree':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
                    'coef0':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
                    'shrinking':[True, False],
                    'probability':[True, False],
                    'tol':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}
        scorer = make_scorer(f1_score)
        svc_obj = RandomizedSearchCV(svc,
                            parameters,
                            scoring = scorer,
                            cv = cv_sets,
                            n_iter=15,
                            verbose=2,
                            n_jobs=-1,
                            random_state= 99)
        svc_fit = svc_obj.fit(self.x_train, self.y_train)
        svc_opt = svc_fit.best_estimator_
        self.model_svc=svc_opt
        self.model_svc.fit(self.x_train, self.y_train)
        pred_svc = self.model_svc.predict(self.x_test)
        lst.append(self.cl_accuracy(self.y_test, pred_svc,'SVC' ,self.model_svc))
        return pd.DataFrame(lst)
    
    def logisticPredict(self,x_predict):
        return self.model_l.predict(x_predict)
    
    def randomForestPredict(self,x_predict):
        return self.model_rf.predict(x_predict)
    
    def svcPredict(self,x_predict):
        return self.model_svm.predict(x_predict)

    def knnPredict(self,x_predict):
        return self.model_knn.predict(x_predict)

    def adaBoostPredict(self,x_predict):
        return self.model_ada.predict(x_predict)
    
    def gradientBoostPredict(self,x_predict):
        return self.model_gb.predict(x_predict)
    
    def XGBoostPredict(self,x_predict):
        return self.model_xgb.predict(x_predict)
    
    def lassoPredict(self,x_predict):
        return self.model_lasso.predict(x_predict)
    
    def SGDCPredict(self,x_predict):
        return self.model_sgdc.predict(x_predict)
    
    def decisionTreePredict(self,x_predict):
        return self.model_dt.predict(x_predict)

    def bernoullinbPredict(self,x_predict):
        return self.model_nb.predict(x_predict)
    
    def multinomialnbPredict(self,x_predict):
        return self.model_mb.predict(x_predict)
    
    def gaussiannbPredict(self,x_predict):
        return self.model_gnb.predict(x_predict)
    
    def baggingPredict(self,x_predict):
        return self.model_bg.predict(x_predict)
    
    def extraTreesPredict(self,x_predict):
        return self.model_xt.predict(x_predict)
    
    def ridgePredict(self,x_predict):
        return self.model_rc.predict(x_predict)


