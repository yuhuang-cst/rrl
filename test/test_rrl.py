# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Yu Huang
# @Email: yuhuang-cst@foxmail.com

import os
import json
from sklearn.model_selection import train_test_split

from rrl import RRLClassifier
from data_reader import DataReader
from rrl.utils import get_cls_metric_dict

DATASET_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')

def run(data_path, info_path, rrl_kwargs):
    reader = DataReader(data_path, info_path, sparse=True)
    reader.read()
    X, y = reader.X, reader.y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    cls = RRLClassifier(**rrl_kwargs)
    cls.fit(X_train, y_train, discrete_features=reader.disc_ids)
    y_test_pred = cls.predict(X_test)
    y_test_score = cls.predict_log_proba(X_test)
    metric_dict = get_cls_metric_dict(y_test, y_test_pred, y_test_score, cls.classes_, cal_auc=True, cal_confusion=True)
    print(json.dumps(metric_dict, indent=2))
    rule_dicts = cls.extract_rules(reader.feature_names, weight_sort=cls.classes_[1], cal_act_rate=True, need_bias=True)
    for rule_dict in rule_dicts:
        print(rule_dict)


if __name__ == '__main__':
    data_path = os.path.join(DATASET_PATH, 'tic-tac-toe.data')
    info_path = os.path.join(DATASET_PATH, 'tic-tac-toe.info')
    rrl_kwargs = {
        'dim_list': [1, 16],
        'batch_size': 32,
        'lr': 0.002,
        'epoch': 401,
        'lr_decay_epoch': 200,
        'weight_decay': 1e-6,
        'verbose': False,
    }
    run(data_path, info_path, rrl_kwargs)


