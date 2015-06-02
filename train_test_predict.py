# -*- coding: utf-8 -*-

import numpy as np
import hickle as hkl
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.cross_validation import cross_val_score

def run(model_name):
    # 訓練データ読み込み
    print "==> loading train data from %s" % (model_name + "_train_(features|labels).hkl")
    train_features = hkl.load(model_name + "_train_features.hkl")
    train_labels = hkl.load(model_name + "_train_labels.hkl")
    print "train_features.shape =", train_features.shape
    print "train_labels.shape =", train_labels.shape

    svm = LinearSVC(C=1.0)
    
    # print "==> training and test"
    # X_train = train_features[-1000:]
    # T_train = train_labels[-1000:]
    # X_test = train_features[:-1000]
    # T_test = train_labels[:-1000]
    # svm.fit(X_train, T_train)
    # Y_test = svm.predict(X_test)
    # print confusion_matrix(T_test, Y_test)
    # print accuracy_score(T_test, Y_test)
    # print classification_report(T_test, Y_test)
    
    # 10分割交差検定
    print "==> cross validation"
    scores = cross_val_score(svm, train_features, train_labels, cv=10)
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std())

    # 全訓練データで学習
    svm.fit(train_features, train_labels)
    
    # テストデータ読み込み
    print "==> loading test data from %s" % (model_name + "_test_(features|labels).hkl")
    test_features = hkl.load(model_name + "_test_features.hkl")
    
    # 予測結果をCSVに書き込む
    print "==> predicting and writing"
    predicted_labels = svm.predict(test_features)
    with open("test.txt") as fr:
        lines = fr.readlines()
    image_ids = []
    for line in lines:
        image_path = line.split()[0]
        image_name = line.split("/")[-1]
        image_id = image_name.split(".")[0]
        image_id = int(image_id)
        image_ids.append(image_id)
    assert len(image_ids) == len(predicted_labels)
    with open(model_name + "_predict.txt", "w") as fw:
        fw.write("id,label\n")
        for i in xrange(len(image_ids)):
            fw.write("%d,%d\n" % (image_ids[i], predicted_labels[i]))
        
if __name__ == "__main__":
    run("alexnet")
    run("vgg16")
    run("googlenet")
