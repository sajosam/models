# import requered libraries for classifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LogisticRegression,RidgeClassifier,SGDClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,classification_report,confusion_matrix
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
    def __init__(self, data_train,data_test):
        self.data_train = data_train
        self.data_test = data_test

    def standard_scaler(self):
        scaler=StandardScaler()
        scaler.fit(self.data_train)
        x_train=scaler.transform(self.data_train)
        x_test=scaler.transform(self.data_test)
        return x_train,x_test
    
    def MinMax(self):
        minmax=MinMaxScaler()
        minmax.fit(self.data_train)
        x_train=minmax.transform(self.data_train)
        x_test=minmax.transform(self.data_test)
        return x_train, x_test


class TrainTestSplit:
        # constructor
    def __init__(self, data_train, data_test, test_size=0.2,random_state=0):
        self.data_train = data_train
        self.data_test = data_test
        self.test_size = test_size
        self.random_state = random_state
    # method for train test split
    def split(self):
        # split the data into training and testing
        x_train, x_test, y_train, y_test = train_test_split(self.data_train, self.data_test, test_size=self.test_size, random_state=self.random_state)
        return x_train, x_test, y_train, y_test


class Modelling:
    # constructor
    def __init__(self, x_train, x_test, y_train, y_test):
        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    # method for train model
    def logistic(self):
        self.model=LogisticRegression()
        self.model.fit(self.X_train, self.y_train)
        y_pred=self.model.predict(self.X_test)
        # accuracy
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        # confusion matrix
        print("Confusion Matrix:", confusion_matrix(self.y_test, y_pred))
        # classification report
        print("Classification Report:", classification_report(self.y_test, y_pred))
        # roc auc score
        print("Roc Auc Score:", roc_auc_score(self.y_test, y_pred))
        return self.model

    def log_pred(self,test):
        y_pred_log=self.model.predict(test)
        return y_pred_log

    def k_nn(self):
        acc_values=[]
        neighbor = np.arange(1,20)
        for k in neighbor:
            modelk=KNeighborsClassifier(n_neighbors = k, metric= 'minkowski')
            modelk.fit(self.X_train,self.y_train)
            y_predict1 = modelk.predict(self.X_test)
            acc=accuracy_score(self.y_test, y_predict1)
            acc_values.append(acc)

        self.model_knn=KNeighborsClassifier(n_neighbors=acc_values.index(max(acc_values)), metric='minkowski')
        self.model_knn.fit(self.X_train, self.y_train)
        y_pred_knn=self.model_knn.predict(self.X_test)
        # accuracy
        print("Accuracy:", accuracy_score(self.y_test, y_pred_knn))
        # confusion matrix
        print("Confusion Matrix:", confusion_matrix(self.y_test, y_pred_knn))
        # classification report
        print("Classification Report:", classification_report(self.y_test, y_pred_knn))
        # roc auc score
        print("Roc Auc Score:", roc_auc_score(self.y_test, y_pred_knn))
        return self.model_knn

    
    def knn_pred(self,test):
        y_pred_knn=self.model_knn.predict(test)
        return y_pred_knn

    def svc(self):
        self.model_svc=SVC(kernel='linear')
        self.model_svc.fit(self.X_train, self.y_train)
        y_pred_svc=self.model_svc.predict(self.X_test)
        # accuracy
        print("Accuracy:", accuracy_score(self.y_test, y_pred_svc))
        # confusion matrix
        print("Confusion Matrix:", confusion_matrix(self.y_test, y_pred_svc))
        # classification report
        print("Classification Report:", classification_report(self.y_test, y_pred_svc))
        # roc auc score
        print("Roc Auc Score:", roc_auc_score(self.y_test, y_pred_svc))
        return self.model_svc

    def svc_pred(self,test):
        y_pred_svc=self.model_svc.predict(test)
        return y_pred_svc


    def decision_tree(self):
        self.model_dt=DecisionTreeClassifier()
        self.model_dt.fit(self.X_train, self.y_train)
        y_pred_dt=self.model_dt.predict(self.X_test)
        # accuracy
        print("Accuracy:", accuracy_score(self.y_test, y_pred_dt))
        # confusion matrix
        print("Confusion Matrix:", confusion_matrix(self.y_test, y_pred_dt))
        # classification report
        print("Classification Report:", classification_report(self.y_test, y_pred_dt))
        # roc auc score
        print("Roc Auc Score:", roc_auc_score(self.y_test, y_pred_dt))
        return self.model_dt

    def dt_pred(self,test):
        y_pred_dt=self.model_dt.predict(test)
        return y_pred_dt


    def random_forest(self):
        self.model_rf=RandomForestClassifier(n_estimators=100)
        self.model_rf.fit(self.X_train, self.y_train)
        y_pred_rf=self.model_rf.predict(self.X_test)
        # accuracy
        print("Accuracy:", accuracy_score(self.y_test, y_pred_rf))
        # confusion matrix
        print("Confusion Matrix:", confusion_matrix(self.y_test, y_pred_rf))
        # classification report
        print("Classification Report:", classification_report(self.y_test, y_pred_rf))
        # roc auc score
        print("Roc Auc Score:", roc_auc_score(self.y_test, y_pred_rf))
        return self.model_rf
    
    def rf_pred(self,test):
        y_pred_rf=self.model_rf.predict(test)
        return y_pred_rf


    def gradient_boosting(self):
        self.model_gb=GradientBoostingClassifier(n_estimators=100)
        self.model_gb.fit(self.X_train, self.y_train)
        y_pred_gb=self.model_gb.predict(self.X_test)
        # accuracy
        print("Accuracy:", accuracy_score(self.y_test, y_pred_gb))
        # confusion matrix
        print("Confusion Matrix:", confusion_matrix(self.y_test, y_pred_gb))
        # classification report
        print("Classification Report:", classification_report(self.y_test, y_pred_gb))
        # roc auc score
        print("Roc Auc Score:", roc_auc_score(self.y_test, y_pred_gb))
        return self.model_gb
    
    def gb_pred(self,test):
        y_pred_gb=self.model_gb.predict(test)
        return y_pred_gb


    def XGBOOST(self):
        self.model_xgb=XGBClassifier(n_estimators=100)
        self.model_xgb.fit(self.X_train, self.y_train)
        y_pred_xgb=self.model_xgb.predict(self.X_test)
        # accuracy
        print("Accuracy:", accuracy_score(self.y_test, y_pred_xgb))
        # confusion matrix
        print("Confusion Matrix:", confusion_matrix(self.y_test, y_pred_xgb))
        # classification report
        print("Classification Report:", classification_report(self.y_test, y_pred_xgb))
        # roc auc score
        print("Roc Auc Score:", roc_auc_score(self.y_test, y_pred_xgb))
        return self.model_xgb
    
    def xgb_pred(self,test):
        y_pred_xgb=self.model_xgb.predict(test)
        return y_pred_xgb

    
    def adaBoostC(self):
        self.model_abc=AdaBoostClassifier(n_estimators=2000,learning_rate=0.1,random_state=42)
        self.model_abc.fit(self.X_train, self.y_train)
        y_pred_abc=self.model_abc.predict(self.X_test)
        # accuracy
        print("Accuracy:", accuracy_score(self.y_test, y_pred_abc))
        # confusion matrix
        print("Confusion Matrix:", confusion_matrix(self.y_test, y_pred_abc))
        # classification report
        print("Classification Report:", classification_report(self.y_test, y_pred_abc))
        # roc auc score
        print("Roc Auc Score:", roc_auc_score(self.y_test, y_pred_abc))
        return self.model_abc
    
    def abc_pred(self,test):
        y_pred_abc=self.model_abc.predict(test)
        return y_pred_abc

    
    def bernoullinb(self):
        self.model_bnb=BernoulliNB()
        self.model_bnb.fit(self.X_train, self.y_train)
        y_pred_bnb=self.model_bnb.predict(self.X_test)
        # accuracy
        print("Accuracy:", accuracy_score(self.y_test, y_pred_bnb))
        # confusion matrix
        print("Confusion Matrix:", confusion_matrix(self.y_test, y_pred_bnb))
        # classification report
        print("Classification Report:", classification_report(self.y_test, y_pred_bnb))
        # roc auc score
        print("Roc Auc Score:", roc_auc_score(self.y_test, y_pred_bnb))
        return self.model_bnb
    
    def bnb_pred(self,test):
        y_pred_bnb=self.model_bnb.predict(test)
        return y_pred_bnb


    def multinomialnb(self):
        self.model_mnb=MultinomialNB()
        self.model_mnb.fit(self.X_train, self.y_train)
        y_pred_mnb=self.model_mnb.predict(self.X_test)
        # accuracy
        print("Accuracy:", accuracy_score(self.y_test, y_pred_mnb))
        # confusion matrix
        print("Confusion Matrix:", confusion_matrix(self.y_test, y_pred_mnb))
        # classification report
        print("Classification Report:", classification_report(self.y_test, y_pred_mnb))
        # roc auc score
        print("Roc Auc Score:", roc_auc_score(self.y_test, y_pred_mnb))
        return self.model_mnb

    def mnb_pred(self,test):
        y_pred_mnb=self.model_mnb.predict(test)
        return y_pred_mnb


    def bagging(self):
        self.model_bagging=BaggingClassifier(n_estimators=100)
        self.model_bagging.fit(self.X_train, self.y_train)
        y_pred_bagging=self.model_bagging.predict(self.X_test)
        # accuracy
        print("Accuracy:", accuracy_score(self.y_test, y_pred_bagging))
        # confusion matrix
        print("Confusion Matrix:", confusion_matrix(self.y_test, y_pred_bagging))
        # classification report
        print("Classification Report:", classification_report(self.y_test, y_pred_bagging))
        # roc auc score
        print("Roc Auc Score:", roc_auc_score(self.y_test, y_pred_bagging))
        return self.model_bagging

    def bagging_pred(self,test):
        y_pred_bagging=self.model_bagging.predict(test)
        return y_pred_bagging

    def extraTrees(self):
        self.model_et=ExtraTreesClassifier(n_estimators=100)
        self.model_et.fit(self.X_train, self.y_train)
        y_pred_et=self.model_et.predict(self.X_test)
        # accuracy
        print("Accuracy:", accuracy_score(self.y_test, y_pred_et))
        # confusion matrix
        print("Confusion Matrix:", confusion_matrix(self.y_test, y_pred_et))
        # classification report
        print("Classification Report:", classification_report(self.y_test, y_pred_et))
        # roc auc score
        print("Roc Auc Score:", roc_auc_score(self.y_test, y_pred_et))
        return self.model_et
    
    def et_pred(self,test):
        y_pred_et=self.model_et.predict(test)
        return y_pred_et

    def ridge(self):
        self.model_r=RidgeClassifier()
        self.model_r.fit(self.X_train, self.y_train)
        y_pred_r=self.model_r.predict(self.X_test)
        # accuracy
        print("Accuracy:", accuracy_score(self.y_test, y_pred_r))
        # confusion matrix
        print("Confusion Matrix:", confusion_matrix(self.y_test, y_pred_r))
        # classification report
        print("Classification Report:", classification_report(self.y_test, y_pred_r))
        # roc auc score
        print("Roc Auc Score:", roc_auc_score(self.y_test, y_pred_r))
        return self.model_r

    def r_pred(self,test):
        y_pred_r=self.model_r.predict(test)
        return y_pred_r

    def sgd(self):
        self.model_sgd=SGDClassifier()
        self.model_sgd.fit(self.X_train, self.y_train)
        y_pred_sgd=self.model_sgd.predict(self.X_test)
        # accuracy
        print("Accuracy:", accuracy_score(self.y_test, y_pred_sgd))
        # confusion matrix
        print("Confusion Matrix:", confusion_matrix(self.y_test, y_pred_sgd))
        # classification report
        print("Classification Report:", classification_report(self.y_test, y_pred_sgd))
        # roc auc score
        print("Roc Auc Score:", roc_auc_score(self.y_test, y_pred_sgd))
        return self.model_sgd

    def sgd_pred(self,test):
        y_pred_sgd=self.model_sgd.predict(test)
        return y_pred_sgd



class classifierSingleModel:

    def __init__(self,X_train,y_train,X_test,y_test):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.model=[LogisticRegression(),KNeighborsClassifier(),GaussianNB(),
                    MultinomialNB(),BaggingClassifier(),ExtraTreesClassifier(),
                    RidgeClassifier(),SGDClassifier(),RandomForestClassifier(),
                    XGBClassifier(),AdaBoostClassifier(),MultinomialNB(),BernoulliNB()]
        self.model_name=['Logistic Regression','KNeighborsClassifier','GaussianNB','MultinomialNB',
                'BaggingClassifier','ExtraTreesClassifier','RidgeClassifier','SGDClassifier',
                'RandomForestClassifier','XGBClassifier','AdaBoostClassifier','MultinomialNB',
                'BernoulliNB']
        self.model_table=[]

    def model_fit(self):
        lst=[]
        for i in range(len(self.model)):
            self.model[i].fit(self.X_train, self.y_train)
            y_pred=self.model[i].predict(self.X_test)
            accuracy=accuracy_score(self.y_test, y_pred)
            confusion=confusion_matrix(self.y_test, y_pred)
            roc=roc_auc_score(self.y_test, y_pred)
            f1=f1_score(self.y_test, y_pred)
            recall=recall_score(self.y_test, y_pred)
            precision=precision_score(self.y_test, y_pred)
            lst.append({'model':self.model_name[i],'accuracy':accuracy,'confusion':confusion,'roc':roc,'f1':f1,'recall':recall,'precision':precision})
        self.model_table=pd.DataFrame(lst)
        return self.model_table


class classifierHyper:
    def __init__(self,x_train,x_test,y_train,y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def adaBoost(self):
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
        print("Accuracy:", accuracy_score(self.y_test,pred_ada))
        print("adaBoost: ", f1_score(self.y_test, pred_ada, average='macro'))
        print("adaBoost: ", self.model_ada.score(self.x_test, self.y_test))
        print("Classification",classification_report(self.y_test, pred_ada))
        return self.model_ada
    
    def ada_pred(self,test):
        pred_ada = self.model_ada.predict(test)
        return pred_ada

    
    def randomforest(self):
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
        print("Accuracy:", accuracy_score(self.y_test,pred_rf))
        print("RandomForest: ", f1_score(self.y_test, pred_rf, average='macro'))
        print("RandomForest: ", self.model_rf.score(self.x_test, self.y_test))
        print("Classification",classification_report(self.y_test, pred_rf))
        return self.model_rf
    
    def rf_pred(self,test):
        pred_rf = self.model_rf.predict(test)
        return pred_rf

    def gradientboost(self):
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
        print("Accuracy:", accuracy_score(self.y_test,pred_gb))
        print("GradientBoost: ", f1_score(self.y_test, pred_gb, average='macro'))
        print("GradientBoost: ", self.model_gb.score(self.x_test, self.y_test))
        print("Classification",classification_report(self.y_test, pred_gb))
        return self.model_gb
    
    def gb_pred(self,test):
        pred_gb = self.model_gb.predict(test)
        return pred_gb
    
    def logistic(self):
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
        print("Accuracy:", accuracy_score(self.y_test,pred_l))
        print("Logistic: ", f1_score(self.y_test, pred_l, average='macro'))
        print("Logistic: ", self.model_l.score(self.x_test, self.y_test))
        print("Classification",classification_report(self.y_test, pred_l))
        return self.model_l

    def l_pred(self,test):
        pred_l = self.model_l.predict(test)
        return pred_l

    def svm(self):
        svm=SVC(random_state=42)
        cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
        parameters = {'C':[0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
                    'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                    'degree':[2,3,4,5],
                    'gamma':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    'coef0':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    'shrinking':[True, False],
                    'probability':[True, False],
                    'tol':[0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    'cache_size':[200, 300, 400, 500, 600, 700, 800, 900, 1000],
                    'class_weight':['balanced', None],
                    'verbose':[0, 1, 2, 3, 4, 5],
                    'decision_function_shape':['ovo', 'ovr'],
                    'break_ties':['random', 'half'],
                    'random_state':[None,1,9,10,11,32,35,42,99],
                    'max_iter':[-1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]}
        scorer = make_scorer(f1_score)
        svm_obj = RandomizedSearchCV(svm,
                parameters,
                scoring = scorer,
                cv = cv_sets,
                verbose=2,
                n_iter=15,
                n_jobs=-1,
                random_state= 99)
        svm_fit = svm_obj.fit(self.x_train, self.y_train)
        svm_opt = svm_fit.best_estimator_
        self.model_svm=svm_opt
        self.model_svm.fit(self.x_train, self.y_train)
        pred_svm = self.model_svm.predict(self.x_test)
        print("Accuracy:", accuracy_score(self.y_test,pred_svm))
        print("SVM: ", f1_score(self.y_test, pred_svm, average='macro'))
        print("SVM: ", self.model_svm.score(self.x_test, self.y_test))
        print("Classification",classification_report(self.y_test, pred_svm))
        return self.model_svm
    
    def svm_pred(self,test):
        pred_svm = self.model_svm.predict(test)
        return pred_svm

    def knn(self):
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
        print("Accuracy:", accuracy_score(self.y_test,pred_knn))
        print("KNN: ", f1_score(self.y_test, pred_knn, average='macro'))
        print("KNN: ", self.model_knn.score(self.x_test, self.y_test))
        print("Classification",classification_report(self.y_test, pred_knn))
        return self.model_knn

    def knn_pred(self,test):
        pred_knn = self.model_knn.predict(test)
        return pred_knn
    
    def binomialnb(self):
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
        print("Accuracy:", accuracy_score(self.y_test,pred_nb))
        print("NB: ", f1_score(self.y_test, pred_nb, average='macro'))
        print("NB: ", self.model_nb.score(self.x_test, self.y_test))
        print("Classification",classification_report(self.y_test, pred_nb))
        return self.model_nb

    def nb_pred(self,test):
        pred_nb = self.model_nb.predict(test)
        return pred_nb


    def multinomialnb(self):
        nb=MultinomialNB()
        cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
        parameters = {'alpha':[0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
            'fit_prior':[True, False],
            'class_prior':[None, [0.1, 0.9]]}
        scorer = make_scorer(f1_score)
        nb_obj = RandomizedSearchCV(nb,
            parameters,
            scoring = scorer,
            cv = cv_sets,
            verbose=2,
            n_iter=15,
            n_jobs=-1,
            random_state= 99)
        nb_fit = nb_obj.fit(self.x_train, self.y_train)
        nb_opt = nb_fit.best_estimator_
        self.model_nb=nb_opt
        self.model_nb.fit(self.x_train, self.y_train)
        pred_nb = self.model_nb.predict(self.x_test)
        print("Accuracy:", accuracy_score(self.y_test,pred_nb))
        print("MultinomialNB: ", f1_score(self.y_test, pred_nb, average='macro'))
        print("MultinomialNB: ", self.model_nb.score(self.x_test, self.y_test))
        print("Classification",classification_report(self.y_test, pred_nb))
        return self.model_nb
    
    def multinomialnb_pred(self,test):
        pred_nb = self.model_nb.predict(test)
        return pred_nb

    def gaussiannb(self):
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
        print("Accuracy:", accuracy_score(self.y_test,pred_gnb))
        print("GaussianNB: ", f1_score(self.y_test, pred_gnb, average='macro'))
        print("GaussianNB: ", self.model_gnb.score(self.x_test, self.y_test))
        print("Classification",classification_report(self.y_test, pred_gnb))
        return self.model_gnb

    def gaussiannb_pred(self,test):
        pred_gnb = self.model_gnb.predict(test)
        return pred_gnb

    def decisiontree(self):
        dt=DecisionTreeClassifier()
        cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
        parameters = {'criterion':['gini', 'entropy'],
                    'splitter':['best', 'random'],
                    'max_depth':[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'min_samples_split':[2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    'min_weight_fraction':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    'max_features':[None, 'auto', 'sqrt', 'log2', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    'random_state':[0,1,10,32,35,40,42,99],
                    'min_impurity_decrease':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    'class_weight':['balanced', None],
                    'presort':[True, False],
                    'max_leaf_nodes':[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'min_samples_leaf':[1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
        scorer = make_scorer(f1_score)
        dt_obj = RandomizedSearchCV(dt,
                                    parameters,
                                    scoring = scorer,
                                    cv = cv_sets,
                                    verbose=2,
                                    n_iter=15,
                                    n_jobs=-1,
                                    random_state= 99)

        dt_fit = dt_obj.fit(self.x_train, self.y_train)
        dt_opt = dt_fit.best_estimator_
        self.model_dt=dt_opt
        self.model_dt.fit(self.x_train, self.y_train)
        pred_dt = self.model_dt.predict(self.x_test)
        print("Accuracy:", accuracy_score(self.y_test,pred_dt))
        print("DecisionTree: ", f1_score(self.y_test, pred_dt, average='macro'))
        print("DecisionTree: ", self.model_dt.score(self.x_test, self.y_test))
        print("Classification",classification_report(self.y_test, pred_dt))
        return self.model_dt
    
    def decisiontree_pred(self,test):
        pred_dt = self.model_dt.predict(test)
        return pred_dt

    def xgboost(self):
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
        print("Accuracy:", accuracy_score(self.y_test,pred_xgb))
        print("XGBoost: ", f1_score(self.y_test, pred_xgb, average='macro'))
        print("XGBoost: ", self.model_xgb.score(self.x_test, self.y_test))
        print("Classification",classification_report(self.y_test, pred_xgb))
        return self.model_xgb
    
    def xgboost_pred(self,test):
        pred_xgb = self.model_xgb.predict(test)
        return pred_xgb
    
    def bagging(self):
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
        print("Accuracy:", accuracy_score(self.y_test,pred_bg))
        print("Bagging: ", f1_score(self.y_test, pred_bg, average='macro'))
        print("Bagging: ", self.model_bg.score(self.x_test, self.y_test))
        print("Classification",classification_report(self.y_test, pred_bg))
        return self.model_bg

    def bagging_pred(self,test):
        pred_bg = self.model_bg.predict(test)
        return pred_bg

    def extra(self):
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
        print("Accuracy:", accuracy_score(self.y_test,pred_xt))
        print("ExtraTrees: ", f1_score(self.y_test, pred_xt, average='macro'))
        print("ExtraTrees: ", self.model_xt.score(self.x_test, self.y_test))
        print("Classification",classification_report(self.y_test, pred_xt))
        return self.model_xt
    
    def extra_pred(self,test):
        pred_xt = self.model_xt.predict(test)
        return pred_xt
    
    def ridge(self):
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
        print("Accuracy:", accuracy_score(self.y_test,pred_rc))
        print("Ridge: ", f1_score(self.y_test, pred_rc, average='macro'))
        print("Ridge: ", self.model_rc.score(self.x_test, self.y_test))
        print("Classification",classification_report(self.y_test, pred_rc))
        return self.model_rc
    
    def ridge_pred(self,test):
        pred_rc = self.model_rc.predict(test)
        return pred_rc




