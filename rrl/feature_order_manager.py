# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Yu Huang
# @Email: yuhuang-cst@foxmail.com

import numpy as np


class FeatureOrderManager(object):
    def __init__(self):
        self.keep_origin = False


    def get_n_feature(self, X):
        if len(X.shape) == 1:
            return X.shape[0]
        assert len(X.shape) == 2
        return X.shape[1]


    def fit(self, X, discrete_features):
        """
        Args:
            X (array-like)
            discrete_features (array-like): A one-dimensional array of categorical columns indices
        """
        n_features = self.get_n_feature(X)
        continous_features = [id for id in range(n_features) if id not in discrete_features]
        self.to_map = np.array(discrete_features + continous_features)
        self.back_map = np.zeros_like(self.to_map)
        self.back_map[self.to_map] = np.arange(n_features)
        if (self.to_map == np.arange(n_features)).all():
            self.keep_origin = True


    def transform(self, X, inplace=True):
        """
        Args:
            X (array-like)
        Returns:
            array-like
        """
        if not inplace:
            X = X.copy()
        if self.keep_origin:
            return X
        if len(X.shape) == 1:
            X[:] = X[self.to_map]
        else:
            assert len(X.shape) == 2
            X[:] = X[:, self.to_map]
        return X


    def fit_transform(self, X, discrete_features, inplace=True):
        self.fit(X, discrete_features)
        return self.transform(X, inplace=inplace)


    def inverse_transform(self, X_prime, inplace=True):
        """
        Args:
            X_prime (array-like)
        Returns:
            array-like
        """
        if not inplace:
            X_prime = X_prime.copy()
        if self.keep_origin:
            return X_prime
        if len(X_prime.shape) == 1:
            X_prime[:] = X_prime[self.back_map]
        else:
            assert len(X_prime.shape) == 2
            X_prime[:] = X_prime[:, self.back_map]
        return X_prime


if __name__ == '__main__':
    manager = FeatureOrderManager()
    feature_names = np.array(['a', 'b', 'c', 'd'])
    X = np.array([[2.2, 0, 1, 3.3], [2.5, 0, 0, 3.5]])
    print('X', X)
    discrete_features = [1, 2]
    X_prime = manager.fit_transform(X, discrete_features)
    print('X_prime', X_prime)
    print('Features of X_prime', manager.transform(feature_names))
    print('X back', manager.inverse_transform(X_prime))