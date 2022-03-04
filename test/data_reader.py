# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Yu Huang
# @Email: yuhuang-cst@foxmail.com

import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DISC_FEAT = 'discrete'
CONTI_FEAT = 'continueous'

class DataReader(object):
    def __init__(self, data_path, info_path, sparse=False, dtype=np.float32):
        self.data_path = data_path
        self.info_path = info_path
        self.sparse = sparse
        self.dtype = dtype

        self.X = None # (n_samples, n_features)
        self.y = None # (n_samples,)
        self.feature_names = None # (n_samples,)
        self.feature_types = None # (n_samples,)
        self.disc_ids = None


    def read_info(self):
        """
        Returns:
            list: [(col_name, col_type), ...]
            int: column id of label in .data file
        """
        with open(self.info_path) as f:
            f_list = []
            for line in f:
                tokens = line.strip().split()
                f_list.append(tokens)
        col_infos, label_col = f_list[:-1], int(f_list[-1][-1])
        if label_col < 0:
            label_col = len(col_infos) + label_col
        return col_infos, label_col


    def read(self):
        df = pd.read_csv(self.data_path)
        col_infos, label_col = self.read_info()
        col_names, col_types = zip(*col_infos)

        df.columns = col_names
        self.y = df.iloc[:, label_col].values
        df = df.drop(df.columns[label_col], axis=1)
        X = df.values

        #         if self.sparse:
        #             X = sp.csr_matrix(np.array(df.values, dtype=np.float64))
        #         else:
        #             X = np.array(df.values, dtype=np.float64)

        feature_types = np.array(col_types[:label_col] + col_types[label_col+1:])
        feature_names = np.array(df.columns)
        disc_ids = [i for i, ft in enumerate(feature_types) if ft == DISC_FEAT]
        conti_ids = [i for i, ft in enumerate(feature_types) if ft == CONTI_FEAT]
        assert len(disc_ids) + len(conti_ids) == len(feature_types)

        X_conti = X[:, conti_ids]
        conti_feature_names = [feature_names[i] for i in conti_ids]
        if conti_ids:
            scaler = StandardScaler()
            X_conti = scaler.fit_transform(X_conti)

        X_disc = X[:, disc_ids]
        if disc_ids:
            enc = OneHotEncoder(categories='auto', drop='first', sparse=False)
            X_disc = enc.fit_transform(X_disc)
            disc_feature_names = enc.get_feature_names([feature_names[id] for id in disc_ids]).tolist()
        else:
            disc_feature_names = []
        self.X = np.hstack([X_disc, X_conti])
        self.X = self.X.astype(self.dtype)
        if self.sparse:
            self.X = sp.csr_matrix(self.X)
        self.feature_names = np.array(disc_feature_names + conti_feature_names)
        self.feature_types = np.array([DISC_FEAT] * len(disc_ids) + [CONTI_FEAT] * len(conti_ids))
        self.disc_ids = list(range(len(disc_feature_names)))


if __name__ == '__main__':
    pass

