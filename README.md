
# Project Title
Machine Learning Models Made Simple




## Guide

### Classfication model
    classifier->
            Scaling-> [train data, test data]
                standard_scaler    
                MinMax
            TrainSplitTest-> [train data, test data, test_size,random_state]
                        split
            classifierSingleModel-> [X_train,y_train,X_test,y_test]
                                model_fit
            classifierHyper-> [x_train,x_test,y_train,y_test]
                        adaBoost, ada_pred 
                        randomforest, rf_pred 
                        gradientboost, gb_pred
                        logistic, l_pred 
                        svm, svm_pred 
                        knn, knn_pred 
                        binomialnb, nb_pred 
                        multinomialnb, multinomialnb_pred 
                        gaussiannb, gaussiannb_pred 
                        decisiontree, decisiontree_pred 
                        xgboost, xgboost_pred 
                        bagging, bagging_pred 
                        extra, extra_pred 
                        ridge, ridge_pred
                                    
            Modelling-> [x_train, x_test, y_train, y_test]
                    logistic, log_pred
                    k_nn, knn_pred 
                    svc, svc_pred 
                    decision_tree, dt_pred 
                    random_forest, rf_pred 
                    gradient_boosting, gb_pred 
                    XGBOOST, xgb_pred 
                    adaBoostC, abc_pred 
                    bernoullinb, bnb_pred 
                    multinomialnb, mnb_pred 
                    bagging, bagging_pred 
                    extraTrees, et_pred 
                    ridge, r_pred 
                    sgd, sgd_pred

### Regression model
    regression->
        regressionSingleModel-> [X_train,y_train,X_test,y_test]
            model_fit
        RegressionHyper-> [x_train,y_train,x_test,y_test]
            lasso, lasso_pred 
            ridge, ridge_pred 
            elastic, elastic_pred 
            svr, svr_pred 
            knn, knn_pred 
            rf, rf_pred 
            gb, gb_pred 
            lr, lr_pred 
            xgboost, xg_pred 
            sgdregressor, sgd_pred 
            decisiontree, dt_pred 
            ada_boost, ada_pred 
            theilsenregressor, theil_pred 
            ransacregressor, ransac_pred  
            orthogonalmatchingpursuit, ortho_pred 
            lassolars, lasso_pred 
            lars, lars_pred 
            huberregresso,r huber_pred 
            passiveaggressiveregressor, passiveaggressive_pred 
            ardregression, ard_pred 
            bayesianridge, bayesianridge_pred 
            baggingregressor, bagging_pred 
            extratreesregressor, extratrees_pred
