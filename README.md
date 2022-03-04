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

# ========= Learned Rules =========
# {'RULE_ID': (-1, 1), 'RULE_DESC': "service_b'http' & src_bytes < 1.8491", 'RULE_PARSE': ('&', ["service_b'http'", 'src_bytes < 1.8491']), '0_WEIGHT': -3.9798991680145264, '0_BIAS': 0.06892254948616028, '1_WEIGHT': 3.4808433055877686, '1_BIAS': 0.1616661697626114, '2_WEIGHT': -2.1388389319181442, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 42945.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.7723504127475136}
# {'RULE_ID': (-1, 13), 'RULE_DESC': "service_b'X11' | service_b'auth' | service_b'domain' | service_b'pop_3' | service_b'smtp' | service_b'ssh' | service_b'telnet'", 'RULE_PARSE': ('|', ["service_b'X11'", "service_b'auth'", "service_b'domain'", "service_b'pop_3'", "service_b'smtp'", "service_b'ssh'", "service_b'telnet'"]), '0_WEIGHT': -0.6997411251068115, '0_BIAS': 0.06892254948616028, '1_WEIGHT': 0.5698036551475525, '1_BIAS': 0.1616661697626114, '2_WEIGHT': -0.30796183086931705, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 7605.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.13677319569087998}
# {'RULE_ID': (-1, 8), 'RULE_DESC': "service_b'X11' | service_b'auth' | service_b'ftp_data' | service_b'other' | service_b'smtp' | service_b'telnet' | src_bytes < 1.8491", 'RULE_PARSE': ('|', ["service_b'X11'", "service_b'auth'", "service_b'ftp_data'", "service_b'other'", "service_b'smtp'", "service_b'telnet'", 'src_bytes < 1.8491']), '0_WEIGHT': -1.1691882610321045, '0_BIAS': 0.06892254948616028, '1_WEIGHT': 0.5673568844795227, '1_BIAS': 0.1616661697626114, '2_WEIGHT': 0.04882293939590454, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 53928.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.969875726129885}
# {'RULE_ID': (-1, 19), 'RULE_DESC': "service_b'X11' | service_b'auth' | service_b'ftp_data' | service_b'other' | service_b'pop_3' | service_b'smtp' | service_b'telnet' | src_bytes < 1.8491", 'RULE_PARSE': ('|', ["service_b'X11'", "service_b'auth'", "service_b'ftp_data'", "service_b'other'", "service_b'pop_3'", "service_b'smtp'", "service_b'telnet'", 'src_bytes < 1.8491']), '0_WEIGHT': -0.8472412824630737, '0_BIAS': 0.06892254948616028, '1_WEIGHT': 0.5277312994003296, '1_BIAS': 0.1616661697626114, '2_WEIGHT': 0.04493652656674385, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 53928.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.969875726129885}
# {'RULE_ID': (-1, 14), 'RULE_DESC': "service_b'X11' | service_b'ftp_data' | service_b'smtp' | service_b'telnet' | src_bytes < 1.8491", 'RULE_PARSE': ('|', ["service_b'X11'", "service_b'ftp_data'", "service_b'smtp'", "service_b'telnet'", 'src_bytes < 1.8491']), '0_WEIGHT': -1.110621452331543, '0_BIAS': 0.06892254948616028, '1_WEIGHT': 0.4573812186717987, '1_BIAS': 0.1616661697626114, '2_WEIGHT': -0.08061549067497253, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 53908.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.9695160333075553}
# {'RULE_ID': (-1, 9), 'RULE_DESC': "service_b'X11' | service_b'ftp_data' | service_b'other' | service_b'smtp' | service_b'telnet' | src_bytes < 1.8491", 'RULE_PARSE': ('|', ["service_b'X11'", "service_b'ftp_data'", "service_b'other'", "service_b'smtp'", "service_b'telnet'", 'src_bytes < 1.8491']), '0_WEIGHT': -1.1531457901000977, '0_BIAS': 0.06892254948616028, '1_WEIGHT': 0.3458557426929474, '1_BIAS': 0.1616661697626114, '2_WEIGHT': -0.11438364535570145, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 53928.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.969875726129885}
# {'RULE_ID': (-1, 20), 'RULE_DESC': "service_b'auth' | service_b'domain' | service_b'ftp' | service_b'http' | service_b'pop_3' | service_b'smtp' | service_b'ssh' | service_b'telnet'", 'RULE_PARSE': ('|', ["service_b'auth'", "service_b'domain'", "service_b'ftp'", "service_b'http'", "service_b'pop_3'", "service_b'smtp'", "service_b'ssh'", "service_b'telnet'"]), '0_WEIGHT': -0.02095118910074234, '0_BIAS': 0.06892254948616028, '1_WEIGHT': 0.14722773432731628, '1_BIAS': 0.1616661697626114, '2_WEIGHT': -0.30179715156555176, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 52738.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.9484740032012661}
# {'RULE_ID': (-1, 16), 'RULE_DESC': "service_b'X11' | service_b'auth' | service_b'domain' | service_b'http' | service_b'other' | service_b'pop_3' | service_b'private' | service_b'smtp' | service_b'ssh' | service_b'telnet'", 'RULE_PARSE': ('|', ["service_b'X11'", "service_b'auth'", "service_b'domain'", "service_b'http'", "service_b'other'", "service_b'pop_3'", "service_b'private'", "service_b'smtp'", "service_b'ssh'", "service_b'telnet'"]), '0_WEIGHT': 0.1519843190908432, '0_BIAS': 0.06892254948616028, '1_WEIGHT': 0.10802769660949707, '1_BIAS': 0.1616661697626114, '2_WEIGHT': -0.22818920016288757, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 52264.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.9399492833120515}
# {'RULE_ID': (-1, 21), 'RULE_DESC': "service_b'X11' | service_b'auth' | service_b'domain' | service_b'pop_3' | service_b'private' | service_b'smtp' | service_b'telnet'", 'RULE_PARSE': ('|', ["service_b'X11'", "service_b'auth'", "service_b'domain'", "service_b'pop_3'", "service_b'private'", "service_b'smtp'", "service_b'telnet'"]), '0_WEIGHT': -0.5675334334373474, '0_BIAS': 0.06892254948616028, '1_WEIGHT': 0.10532334446907043, '1_BIAS': 0.1616661697626114, '2_WEIGHT': -0.21844902634620667, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 7605.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.13677319569087998}
# {'RULE_ID': (-1, 12), 'RULE_DESC': "service_b'auth' | service_b'domain' | service_b'pop_3' | service_b'smtp' | service_b'telnet'", 'RULE_PARSE': ('|', ["service_b'auth'", "service_b'domain'", "service_b'pop_3'", "service_b'smtp'", "service_b'telnet'"]), '0_WEIGHT': -0.3235155940055847, '0_BIAS': 0.06892254948616028, '1_WEIGHT': 0.0856117233633995, '1_BIAS': 0.1616661697626114, '2_WEIGHT': -0.2563599944114685, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 7601.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.13670125712641404}
# {'RULE_ID': (-1, 4), 'RULE_DESC': 'src_bytes > -0.9307 & src_bytes < 1.8491', 'RULE_PARSE': ('&', ['src_bytes > -0.9307', 'src_bytes < 1.8491']), '0_WEIGHT': -1.9052687883377075, '0_BIAS': 0.06892254948616028, '1_WEIGHT': 0.07451263070106506, '1_BIAS': 0.1616661697626114, '2_WEIGHT': 0.6225391030311584, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 52281.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.9402550222110317}
# {'RULE_ID': (-1, 0), 'RULE_DESC': 'src_bytes > -0.9307 & dst_bytes > 0.3319 & src_bytes < 1.8491 & dst_bytes < 1.3618', 'RULE_PARSE': ('&', ['src_bytes > -0.9307', 'dst_bytes > 0.3319', 'src_bytes < 1.8491', 'dst_bytes < 1.3618']), '0_WEIGHT': -1.103479027748108, '0_BIAS': 0.06892254948616028, '1_WEIGHT': -0.0002398513024672866, '1_BIAS': 0.1616661697626114, '2_WEIGHT': 0.6330946683883667, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 17749.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.31920939517651925}
# {'RULE_ID': (-1, 18), 'RULE_DESC': "service_b'X11' | service_b'auth' | service_b'domain' | service_b'http' | service_b'pop_3' | service_b'smtp' | service_b'ssh' | service_b'telnet'", 'RULE_PARSE': ('|', ["service_b'X11'", "service_b'auth'", "service_b'domain'", "service_b'http'", "service_b'pop_3'", "service_b'smtp'", "service_b'ssh'", "service_b'telnet'"]), '0_WEIGHT': 0.6269895434379578, '0_BIAS': 0.06892254948616028, '1_WEIGHT': -0.009439049288630486, '1_BIAS': 0.1616661697626114, '2_WEIGHT': -0.9082053899765015, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 52224.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.9392298976673921}
# {'RULE_ID': (-1, 7), 'RULE_DESC': 'src_bytes > -0.9307 & dst_bytes > 0.3319 & src_bytes < 1.8491', 'RULE_PARSE': ('&', ['src_bytes > -0.9307', 'dst_bytes > 0.3319', 'src_bytes < 1.8491']), '0_WEIGHT': -0.9839613437652588, '0_BIAS': 0.06892254948616028, '1_WEIGHT': -0.0160017479211092, '1_BIAS': 0.1616661697626114, '2_WEIGHT': 0.38711076974868774, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 18731.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.336870312752909}
# {'RULE_ID': (-1, 17), 'RULE_DESC': "service_b'auth' | service_b'http' | service_b'pop_3' | service_b'private' | service_b'smtp' | service_b'ssh' | service_b'telnet'", 'RULE_PARSE': ('|', ["service_b'auth'", "service_b'http'", "service_b'pop_3'", "service_b'private'", "service_b'smtp'", "service_b'ssh'", "service_b'telnet'"]), '0_WEIGHT': 0.30729344487190247, '0_BIAS': 0.06892254948616028, '1_WEIGHT': -0.07110020518302917, '1_BIAS': 0.1616661697626114, '2_WEIGHT': -0.5806816220283508, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 52219.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.9391399744618096}
# {'RULE_ID': (-1, 10), 'RULE_DESC': "service_b'http'", 'RULE_PARSE': ('|', ["service_b'http'"]), '0_WEIGHT': 0.4952046573162079, '0_BIAS': 0.06892254948616028, '1_WEIGHT': -0.10162308067083359, '1_BIAS': 0.1616661697626114, '2_WEIGHT': -0.10743280500173569, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 44619.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.8024567019765121}
# {'RULE_ID': (-1, 6), 'RULE_DESC': "service_b'http' & dst_bytes > 0.3319", 'RULE_PARSE': ('&', ["service_b'http'", 'dst_bytes > 0.3319']), '0_WEIGHT': 1.0337684154510498, '0_BIAS': 0.06892254948616028, '1_WEIGHT': -0.22538410127162933, '1_BIAS': 0.1616661697626114, '2_WEIGHT': -0.6256212592124939, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 19949.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.3587756056327896}
# {'RULE_ID': (-1, 11), 'RULE_DESC': "service_b'http' | service_b'telnet'", 'RULE_PARSE': ('|', ["service_b'http'", "service_b'telnet'"]), '0_WEIGHT': 0.5014243721961975, '0_BIAS': 0.06892254948616028, '1_WEIGHT': -0.28079041838645935, '1_BIAS': 0.1616661697626114, '2_WEIGHT': -0.3636201322078705, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 44777.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.805298275272917}
# {'RULE_ID': (-1, 22), 'RULE_DESC': "service_b'ftp' | service_b'ftp_data'", 'RULE_PARSE': ('|', ["service_b'ftp'", "service_b'ftp_data'"]), '0_WEIGHT': -0.25907427072525024, '0_BIAS': 0.06892254948616028, '1_WEIGHT': -0.3488233983516693, '1_BIAS': 0.1616661697626114, '2_WEIGHT': 0.6701340079307556, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 3315.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.05961908530115281}
# {'RULE_ID': (-1, 5), 'RULE_DESC': "service_b'ftp_data' & src_bytes > -0.9307 & src_bytes < 1.8491", 'RULE_PARSE': ('&', ["service_b'ftp_data'", 'src_bytes > -0.9307', 'src_bytes < 1.8491']), '0_WEIGHT': -0.28408095240592957, '0_BIAS': 0.06892254948616028, '1_WEIGHT': -0.3997609317302704, '1_BIAS': 0.1616661697626114, '2_WEIGHT': 0.691598117351532, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 1753.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.03152707587720087}
# {'RULE_ID': (-1, 15), 'RULE_DESC': "service_b'http' | service_b'ssh'", 'RULE_PARSE': ('|', ["service_b'http'", "service_b'ssh'"]), '0_WEIGHT': 0.8201107382774353, '0_BIAS': 0.06892254948616028, '1_WEIGHT': -0.42447254061698914, '1_BIAS': 0.1616661697626114, '2_WEIGHT': -0.4861374795436859, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 44620.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.8024746866176286}
# {'RULE_ID': (-1, 3), 'RULE_DESC': "service_b'http' & src_bytes > -0.9307 & dst_bytes > 0.3319 & dst_bytes < 1.3618", 'RULE_PARSE': ('&', ["service_b'http'", 'src_bytes > -0.9307', 'dst_bytes > 0.3319', 'dst_bytes < 1.3618']), '0_WEIGHT': 1.61014986038208, '0_BIAS': 0.06892254948616028, '1_WEIGHT': -0.4469049274921417, '1_BIAS': 0.1616661697626114, '2_WEIGHT': -0.9725744128227234, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 19048.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.34257144398683526}
# {'RULE_ID': (-1, 2), 'RULE_DESC': "service_b'ftp' & src_bytes > -0.9307 & dst_bytes > 0.3319 & dst_bytes < 1.3618", 'RULE_PARSE': ('&', ["service_b'ftp'", 'src_bytes > -0.9307', 'dst_bytes > 0.3319', 'dst_bytes < 1.3618']), '0_WEIGHT': 0.5451915860176086, '0_BIAS': 0.06892254948616028, '1_WEIGHT': -0.9348108768463135, '1_BIAS': 0.1616661697626114, '2_WEIGHT': 0.9681420922279358, '2_BIAS': -0.2883208692073822, 'ACT_NODE': 301.0, 'TOTAL_NODE': 55603.0, 'ACT_RATE': 0.005413376976062443}
# RRL:
#                precision    recall  f1-score   support
#
#            0       1.00      1.00      1.00       433
#            1       1.00      0.98      0.99     13992
#            2       0.35      0.89      0.50       208
#
#     accuracy                           0.97     14633
#    macro avg       0.78      0.95      0.83     14633
# weighted avg       0.99      0.97      0.98     14633
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


