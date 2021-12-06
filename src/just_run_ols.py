import os
os.chdir('src/')
import numpy as np
import pandas as pd
from functions import *

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, RepeatedKFold

from sklearn.svm import SVC 
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier


# load data 
X_train = pd.read_csv('../data/X_train.csv', index_col='id')
X_test = pd.read_csv('../data/X_test.csv', index_col='id')
y_train = pd.read_csv('../data/y_train.csv', index_col='id')

# create features
p = 5 # lag of AR process
features_train = feature_process(X_train, p)
features_train.to_csv('../out/features_train.csv', index=False)
features_test = feature_process(X_test, p)
features_test.to_csv('../out/features_test.csv', index=False)


# fill nan/drop nan
features_train = pd.read_csv('../out/features_train.csv')
# features_train = features_train.fillna(0)
features_train = features_train.fillna(features_train.median())
# y_train = y_train.iloc[features_train.index]

features_train.columns = range(len(features_train.columns))

# baseline 
from xgboost import XGBClassifier
model_baseline = XGBClassifier(n_estimators=1000, eta = 0.1, objective='multi:softmax' , use_label_encoder=False, eval_metric='mlogloss') # LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=42)
baseline = np.mean(cross_val_score(model_baseline, features_train, np.ravel(y_train), scoring="f1_micro", cv=crossvalidation, n_jobs=-1))
print("A baseline XGBoost model achieves an F1 score of: ", baseline)


# engineer based on multinomial logit
# print("feature engineering")
# model_engineering = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000)
# path = '../out/features_train_engineered.csv'
# features_train_engineered = feature_engineering(features_train, np.ravel(y_train), model_engineering, 5, 3, 'f1_macro', -1, 10, path)


# # apply engineering to test data
# features_train_engineered = pd.read_csv('../out/features_train_engineered.csv')
# path = '../out/features_test_engineered.csv'
# features_test_engineered = engineered_testdata(features_train_engineered, features_train_engineered.columns, path)

#the models that were tuned individually
models = list()
models.append(('xgb', XGBClassifier(objective='multi:softmax', num_class = 4, use_label_encoder=False, n_estimators = 1000, eta = 0.01, gamma = 0.5, min_child_weight = 5, subsample = 0.75, max_depth = 5)))
models.append(('svm', SVC(C=12, gamma = 0.01, kernel = "rbf")))
#the meta model to combine the individual predictions
meta_model = GradientBoostingClassifier()

#the way this works is is fit the "models" on the full x
#then best parameters for the meta model are found using cross validation
cv = RepeatedKFold(n_splits=3, n_repeats=1, random_state=42) 
stack_model = StackingClassifier(estimators=models, final_estimator=meta_model, cv = cv)
stack_model.fit(X_train, y_train) #import to enter X as z-scores here since svm is sensitive to that

