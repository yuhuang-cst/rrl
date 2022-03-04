# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Yu Huang
# @Email: yuhuang-cst@foxmail.com

import numpy as np
from sklearn.datasets import load_iris, load_boston, fetch_kddcup99
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from rrl import RRLClassifier

dataset = fetch_kddcup99(subset='SF')
X, y = dataset.data, dataset.target
kept_rows = [i for i, label in enumerate(y) if label == b'normal.' or label == b'back.' or label == b'warezclient.']
X, y = X[kept_rows], y[kept_rows]
label_enc = LabelEncoder()
y = label_enc.fit_transform(y)

enc = OneHotEncoder(categories='auto', drop='first', sparse=False)
X_disc = enc.fit_transform(X[:, 1:2])
feature_names_disc = enc.get_feature_names([dataset.feature_names[1]])

X_conti = X[:, [0, 2, 3]]
scaler_conti = StandardScaler()
X_conti = scaler_conti.fit_transform(X_conti)
feature_names_conti = np.array(dataset.feature_names)[[0, 2, 3]]

X = np.hstack([X_disc, X_conti]).astype(np.float64)
feature_names = np.hstack([feature_names_disc, feature_names_conti])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# RRL
cls = RRLClassifier(dim_list=[1, 16], lr=0.01, batch_size=32, epoch=5, eval_metric='macro avg f1-score', verbose=True)
cls.fit(X_train, y_train, discrete_features=list(range(len(feature_names_disc))))
y_test_pred = cls.predict(X_test)
print('========= Learned Rules =========')
rule_dicts = cls.extract_rules(feature_names, weight_sort=cls.classes_[1], cal_act_rate=True, need_bias=True)
for rule_dict in rule_dicts:
    print(rule_dict)
print('RRL:\n', classification_report(y_test, y_test_pred))

# Logistic Regression
from sklearn.linear_model import LogisticRegression
cls = LogisticRegression()
cls.fit(X_train, y_train)
y_test_pred = cls.predict(X_test)
print('LR:\n', classification_report(y_test, y_test_pred))



