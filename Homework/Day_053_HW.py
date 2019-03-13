#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer

df_train = pd.read_csv('Day_053_train_data.csv').fillna(0)
df_train.set_index(['name'], inplace=True)
df_train = df_train.drop(['email_address'], axis=1)
df_test = pd.read_csv('test_features.csv').fillna(0)
df_test.set_index(['name'], inplace=True)
df_test = df_test.drop(['email_address'], axis=1)

drop_feature = ['total_payments', 'total_stock_value']
y_train = df_train['poi']
X_train = df_train.drop(['poi']+drop_feature, axis=1)
X_test = df_test.drop(drop_feature, axis=1)

# scale X_train and X_test
for col in X_train.columns:
    scaler = Normalizer()
    scaler.fit(X_train[col].values.reshape(-1,1))
    X_train[col] = scaler.transform(X_train[col].values.reshape(-1,1))
    X_test[col] = scaler.transform(X_test[col].values.reshape(-1,1))

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_knn_pred = knn.predict_proba(X_test)

# get y_score
y_score = y_knn_pred[:,1]
df_submit = pd.DataFrame({'name':df_test.index, 'poi':y_score})
df_submit.to_csv('submission.csv', index=None)
