import os
os.chdir('src/')

from functions import *
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# load data 
X_train = pd.read_csv('../data/X_train.csv', index_col='id')
X_test = pd.read_csv('../data/X_test.csv', index_col='id')
y_train = pd.read_csv('../data/y_train.csv', index_col='id')
X = pd.concat([X_train, X_test]).reset_index()

# create features
p = 5 # lag of AR process
features = feature_process(X, p)
features_test = features.iloc[0:len(X)]
features_test.to_csv('../out/features_test.csv', index=False)
features_train = features.iloc[len(X_train):len(X)]
features_train.to_csv('../out/features_test.csv', index=False)


# engineer based on multinomial logit
print("feature engineering")
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
path = '../out/X_train_engineered.csv'
features_test_engineered = feature_engineering(features_train, y_train, model, 10, 3, 'f1_macro', -1, 10, path)
features_test_engineered.to_csv('../out/features_test_engineered.csv', index=False)