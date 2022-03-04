# RRL
This is an implementation of **RRL** (Rule-based Representation Learner) with sklearn-like interfaces.

Reference: 
[1] Zhuo Wang et al. (NeurIPS 2021). Scalable Rule-Based Representation Learning for Interpretable Classiﬁcation.

## Installation
```
pip3 install git+https://github.com/yuhuang-cst/rrl.git
```

## Usage

```python
from rrl import RRLClassifier

cls = RRLClassifier()
cls.fit(X, y, discrete_features=[0, 1])
y_pred = cls.predict(X)
```

## Demo
```python
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

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
cls = RRLClassifier(dim_list=[1, 8, 8], lr=0.01, batch_size=64, epoch=5, eval_metric='macro avg f1-score', verbose=True, save_folder='./rrl_kdd')
cls.fit(X_train, y_train, discrete_features=list(range(len(feature_names_disc))))
y_test_pred = cls.predict(X_test)
print('========= 5 Learned Rules =========')
rule_dicts = cls.extract_rules(feature_names, weight_sort=cls.classes_[1], cal_act_rate=True, need_bias=True)
for rule_dict in rule_dicts[:5]:
    print(rule_dict)
print('RRL:\n', classification_report(y_test, y_test_pred))

# Logistic Regression
from sklearn.linear_model import LogisticRegression
cls = LogisticRegression()
cls.fit(X_train, y_train)
y_test_pred = cls.predict(X_test)
print('LR:\n', classification_report(y_test, y_test_pred))

# ========= 5 Learned Rules =========
# {'RULE_ID': (-2, 0), 'RULE_DESC': "service_b'http' & src_bytes < 2.5357", 'RULE_PARSE': ('&', ["service_b'http'", 'src_bytes < 2.5357']), '0_WEIGHT': -3.1222150325775146, '0_BIAS': 0.32828742265701294, '1_WEIGHT': 2.708388864994049, '1_BIAS': -0.10009676963090897, '2_WEIGHT': -1.8755800426006317, '2_BIAS': -0.07201618701219559, 'ACT_NODE': 42917.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.771846842796252}
# {'RULE_ID': (-1, 8), 'RULE_DESC': "(service_b'http' & src_bytes < 2.5357) | (service_b'X11' | service_b'auth' | service_b'domain' | service_b'pop_3' | service_b'private' | service_b'smtp' | service_b'telnet')", 'RULE_PARSE': ('|', [('&', ["service_b'http'", 'src_bytes < 2.5357']), ('|', ["service_b'X11'", "service_b'auth'", "service_b'domain'", "service_b'pop_3'", "service_b'private'", "service_b'smtp'", "service_b'telnet'"])]), '0_WEIGHT': -2.487107992172241, '0_BIAS': 0.32828742265701294, '1_WEIGHT': 1.4219193011522293, '1_BIAS': -0.10009676963090897, '2_WEIGHT': 0.01616627350449562, '2_BIAS': -0.07201618701219559, 'ACT_NODE': 50554.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.9091955470028595}
# {'RULE_ID': (-2, 6), 'RULE_DESC': "service_b'X11' | service_b'auth' | service_b'domain' | service_b'ftp_data' | service_b'other' | service_b'pop_3' | service_b'smtp' | service_b'telnet' | duration > 2.5743 | dst_bytes < -1.9975", 'RULE_PARSE': ('|', ["service_b'X11'", "service_b'auth'", "service_b'domain'", "service_b'ftp_data'", "service_b'other'", "service_b'pop_3'", "service_b'smtp'", "service_b'telnet'", 'duration > 2.5743', 'dst_bytes < -1.9975']), '0_WEIGHT': -0.5070949792861938, '0_BIAS': 0.32828742265701294, '1_WEIGHT': 0.6655272841453552, '1_BIAS': -0.10009676963090897, '2_WEIGHT': -0.5626970529556274, '2_BIAS': -0.07201618701219559, 'ACT_NODE': 11103.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.19968347031634984}
# {'RULE_ID': (-1, 4), 'RULE_DESC': "(src_bytes > -2.8730 & dst_bytes > 0.0448) & (service_b'auth' | service_b'ftp_data' | service_b'smtp' | service_b'telnet' | src_bytes < 2.5357) & (service_b'auth' | service_b'domain' | service_b'http' | service_b'other' | service_b'pop_3' | service_b'smtp' | service_b'telnet')", 'RULE_PARSE': ('&', [('&', ['src_bytes > -2.8730', 'dst_bytes > 0.0448']), ('|', ["service_b'auth'", "service_b'ftp_data'", "service_b'smtp'", "service_b'telnet'", 'src_bytes < 2.5357']), ('|', ["service_b'auth'", "service_b'domain'", "service_b'http'", "service_b'other'", "service_b'pop_3'", "service_b'smtp'", "service_b'telnet'"])]), '0_WEIGHT': -0.530015230178833, '0_BIAS': 0.32828742265701294, '1_WEIGHT': 0.5613583922386169, '1_BIAS': -0.10009676963090897, '2_WEIGHT': -0.7758337259292603, '2_BIAS': -0.07201618701219559, 'ACT_NODE': 28057.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.5045950758052623}
# {'RULE_ID': (-2, 5), 'RULE_DESC': "service_b'ftp_data' | service_b'smtp' | service_b'telnet' | src_bytes < 2.5357 | dst_bytes < -1.9975", 'RULE_PARSE': ('|', ["service_b'ftp_data'", "service_b'smtp'", "service_b'telnet'", 'src_bytes < 2.5357', 'dst_bytes < -1.9975']), '0_WEIGHT': -0.9938647747039795, '0_BIAS': 0.32828742265701294, '1_WEIGHT': 0.47980478405952454, '1_BIAS': -0.10009676963090897, '2_WEIGHT': 0.23347823321819305, '2_BIAS': -0.07201618701219559, 'ACT_NODE': 53917.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.9696778950776037}
# RRL:
#                precision    recall  f1-score   support
# 
#            0       1.00      1.00      1.00       433
#            1       0.99      1.00      0.99     13992
#            2       0.64      0.41      0.50       208
# 
#     accuracy                           0.99     14633
#    macro avg       0.87      0.80      0.83     14633
# weighted avg       0.99      0.99      0.99     14633
# 
# LR:
#                precision    recall  f1-score   support
# 
#            0       1.00      1.00      1.00       433
#            1       0.99      1.00      0.99     13992
#            2       0.09      0.01      0.02       208
# 
#     accuracy                           0.98     14633
#    macro avg       0.69      0.67      0.67     14633
# weighted avg       0.97      0.98      0.98     14633
```


