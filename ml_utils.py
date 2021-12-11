import time
from sklearn.model_selection import GridSearchCV
def random_forest_model(X_train, y_train, cross_validation, n_splits, param_search, scoring):
    from sklearn.ensemble import RandomForestClassifier
    start_time = time.time()
    #Define model
    model = RandomForestClassifier()

    if cross_validation:
        gsearch = GridSearchCV(estimator=model, cv=n_splits,
                               scoring=scoring,
                               param_grid=param_search).fit(X_train.values, y_train.values)

        # Best Model
        best_model = RandomForestClassifier(**gsearch.best_params_).fit(X_train.values, y_train.values)
        print("Best params are {}".format(gsearch.best_params_))
    else:
        best_params = {'max_depth': 15,
                       'max_features': 0.7,
                       'n_estimators': 500,
                       'random_state': 42,
                       'min_samples_leaf': 5,
                       }

        print("Best params are {}".format(best_params))
        best_model = RandomForestClassifier(**best_params).fit(X_train.values, y_train.values)
    print("Train score is %s" % (best_model.score(X_train.values, y_train.values)))
    print(time.time() - start_time)
    return best_model

def xgb_model(X_train, y_train,cross_validation,n_splits,param_search,scoring):
    from xgboost import XGBClassifier
    start_time = time.time()
    model = XGBClassifier()

    if cross_validation:
        gsearch = GridSearchCV(estimator=model, cv=n_splits,
                               scoring=scoring,
                               param_grid=param_search).fit(X_train.values, y_train.values)

        # Best Model
        best_model = XGBClassifier(**gsearch.best_params_).fit(X_train.values, y_train.values)
        print("Best params are {}".format(gsearch.best_params_))

    else:
        best_params = {'max_depth': 4,
                       'colsample_bytree': 0.7,
                       'n_estimators': 500,
                       'random_state': 42,
                       'learning_rate': 0.1,
                       'min_child_weight': 10
                       }

        print("Best params are {}".format(best_params))
        best_model = XGBClassifier(**best_params).fit(X_train.values, y_train.values)
    print("Train score is %s" % (best_model.score(X_train.values, y_train.values)))
    print(time.time() - start_time)
    return best_model

def lr_model(X_train, y_train,cross_validation,n_splits,param_search,scoring,scaled = True):
    from sklearn.linear_model import LogisticRegression
    model=LogisticRegression()

    start_time = time.time()
    # if scaled:
    #     from sklearn.preprocessing import MinMaxScaler
    #     scaler=MinMaxScaler()
    #     X_train_scaled=scaler.fit_transform(X_train)
    #     from sklearn.preprocessing import MinMaxScaler
    #     X_test_scaled=scaler.transform(X_test)



    if cross_validation:
        if scaled:
            gsearch = GridSearchCV(estimator=model, cv=n_splits,
                               scoring=scoring,
                               param_grid=param_search).fit(X_train, y_train)

            # Best Model
            best_model = LogisticRegression(**gsearch.best_params_).fit(X_train, y_train)
            print("Best params are {}".format(gsearch.best_params_))
        else:
            gsearch = GridSearchCV(estimator=model, cv=n_splits,
                                scoring=scoring,
                                param_grid=param_search).fit(X_train.values, y_train.values)

            # Best Model
            best_model = LogisticRegression(**gsearch.best_params_).fit(X_train.values, y_train.values)
            print("Best params are {}".format(gsearch.best_params_))

    else:
        best_params = {'C': 1
                       }

        print("Best params are {}".format(best_params))
        best_model = LogisticRegression(**best_params).fit(X_train.values, y_train.values)
    #print("Train score is %s" % (best_model.score(X_train.values, y_train.values)))
    print(time.time() - start_time)
    return best_model
