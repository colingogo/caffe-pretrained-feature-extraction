# -*- coding: utf-8 -*-

import numpy as np
import hickle as hkl

if __name__ == "__main__":
    train_features = hkl.load("googlenet_train_features.hkl")
    xs = []
    for x in train_features:
        x = x[:,0,0]
        xs.append(x)
    xs = np.asarray(xs)
    hkl.dump(xs, "googlenet_train_features_tmp.hkl", mode="w")

    test_features = hkl.load("googlenet_test_features.hkl")
    xs = []
    for x in test_features:
        x = x[:,0,0]
        xs.append(x)
    xs = np.asarray(xs)
    hkl.dump(xs, "googlenet_test_features_tmp.hkl", mode="w")
