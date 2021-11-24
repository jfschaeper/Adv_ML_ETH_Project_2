import os
os.chdir('src/')

from functions import *
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# load data 
X_train = pd.read_csv('../data/X_train.csv', index_col='id')
X_test = pd.read_csv('../data/X_test.csv', index_col='id')
y_train = pd.read_csv('../data/y_train.csv', index_col='id')
X = pd.concat([X_train, X_test]).reset_index()


# create features
p = 5 # lag of AR process
features_train = feature_process(X_train, p)
features_train.to_csv('../out/features_train.csv', index=False)
# features_train = features.iloc[0:len(X_train)]
features_train.to_csv('../out/features_train.csv', index=False)
features_test = features.iloc[len(X_train):len(X)]
features_test.to_csv('../out/features_test.csv', index=False)

features_test = feature_process(X_test, p)
features_test.to_csv('../out/features_test.csv', index=False)


# fill nan/drop nan
features_train = pd.read_csv('../out/features_train.csv')
# features_train = features_train.fillna(0)
features_train = features_train.fillna(features_train.median())
# y_train = y_train.iloc[features_train.index]


# baseline 
from xgboost import XGBClassifier
model_baseline = XGBClassifier(n_estimators=1000, eta = 0.1, objective='multi:softmax' , use_label_encoder=False, eval_metric='mlogloss') # LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=42)
baseline = np.mean(cross_val_score(model_baseline, features_train, np.ravel(y_train), scoring="f1_micro", cv=crossvalidation, n_jobs=-1))
print("A baseline multinomial logistic regression model achieves an F1 score of: ", baseline)


# engineer based on multinomial logit
print("feature engineering")
model_engineering = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000)
path = '../out/features_train_engineered.csv'
features_train_engineered = feature_engineering(features_train, np.ravel(y_train), model_engineering, 5, 3, 'f1_macro', -1, 10, path)


# apply engineering to test data
features_train_engineered = pd.read_csv('../out/features_train_engineered.csv')
path = '../out/features_test_engineered.csv'
features_test_engineered = engineered_testdata(features_train_engineered, features_train_engineered.columns, path)