def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import Funcs as f
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

ds1, ds2, ds3, labels, folds = f.data_reader()
print("Data preparation is done!")

feat_nums = [50,100,150,200,250]

Data1 = [f.fsf(ds1, labels, n) for n in feat_nums]
Data2 = [f.fsf(ds2, labels, n) for n in feat_nums]
Data3 = [f.fsf(ds3, labels, n) for n in feat_nums]




#create the tables
data1_svm = {50 : f.pd_creator(), 100: f.pd_creator(), 150: f.pd_creator(),200: f.pd_creator(),250: f.pd_creator()}
data2_svm = {50 : f.pd_creator(), 100: f.pd_creator(), 150: f.pd_creator(),200: f.pd_creator(),250: f.pd_creator()}
data3_svm = {50 : f.pd_creator(), 100: f.pd_creator(), 150: f.pd_creator(),200: f.pd_creator(),250: f.pd_creator()}

data1_LG = {50 : f.pd_creator(), 100: f.pd_creator(), 150: f.pd_creator(),200: f.pd_creator(),250: f.pd_creator()}
data2_LG = {50 : f.pd_creator(), 100: f.pd_creator(), 150: f.pd_creator(),200: f.pd_creator(),250: f.pd_creator()}
data3_LG = {50 : f.pd_creator(), 100: f.pd_creator(), 150: f.pd_creator(),200: f.pd_creator(),250: f.pd_creator()}

for fold in range(10):
    print("fold = ",fold+1)
    train = np.where(folds[:,fold] == 0)[0]
    test = np.where(folds[:,fold] == 1)[0]

    for d in range(len(feat_nums)):
        print('number of features : ',feat_nums[d])

        x_train1 = Data1[d][train , :]
        x_train2 = Data2[d][train , :]
        x_train3 = Data3[d][train , :]

        y_train1 = labels[train]
        y_train2 = labels[train]
        y_train3 = labels[train]

        x_test1 = Data1[d][test, :]
        x_test2 = Data2[d][test, :]
        x_test3 = Data3[d][test, :]

        y_test1 = labels[test]
        y_test2 = labels[test]
        y_test3 = labels[test]

        svm_classifier1 = svm.LinearSVC().fit(x_train1, y_train1)
        svm_classifier2 = svm.LinearSVC().fit(x_train2, y_train2)
        svm_classifier3 = svm.LinearSVC().fit(x_train3, y_train3)

        LG_classifier1 = LogisticRegression().fit(x_train1, y_train1)
        LG_classifier2 = LogisticRegression().fit(x_train2, y_train2)
        LG_classifier3 = LogisticRegression().fit(x_train3, y_train3)

        svm_y1_p = svm_classifier1.predict(x_test1)
        svm_y2_p = svm_classifier2.predict(x_test2)
        svm_y3_p = svm_classifier3.predict(x_test3)

        LG_y1_p = LG_classifier1.predict(x_test1)
        LG_y2_p = LG_classifier2.predict(x_test2)
        LG_y3_p = LG_classifier3.predict(x_test3)

        print("SVM & data1")
        ACC, PPV, NPV, recall, TNR, f1, mcc, TP, FP, TN, FN = f.printer(y_test1,svm_y1_p)
        data1_svm[feat_nums[d]] = f.pd_updater(ACC, PPV, NPV, recall, TNR, f1, mcc, TP, FP, TN, FN,
                                               "Fold-{}".format(str(fold+1)),data1_svm[feat_nums[d]])

        print("SVM & data2")
        ACC, PPV, NPV, recall, TNR, f1, mcc, TP, FP, TN, FN = f.printer(y_test2, svm_y2_p)
        data2_svm[feat_nums[d]] = f.pd_updater(ACC, PPV, NPV, recall, TNR, f1, mcc, TP, FP, TN, FN,
                                               "Fold-{}".format(str(fold + 1)), data1_svm[feat_nums[d]])
        print("SVM & data3")
        ACC, PPV, NPV, recall, TNR, f1, mcc, TP, FP, TN, FN = f.printer(y_test3, svm_y3_p)
        data3_svm[feat_nums[d]] = f.pd_updater(ACC, PPV, NPV, recall, TNR, f1, mcc, TP, FP, TN, FN,
                                               "Fold-{}".format(str(fold + 1)), data1_svm[feat_nums[d]])

        print("LogisticRegression & data1")
        ACC, PPV, NPV, recall, TNR, f1, mcc, TP, FP, TN, FN = f.printer(y_test1, LG_y1_p)
        data1_LG[feat_nums[d]] = f.pd_updater(ACC, PPV, NPV, recall, TNR, f1, mcc, TP, FP, TN, FN,
                                               "Fold-{}".format(str(fold + 1)), data1_svm[feat_nums[d]])

        print("LogisticRegression & data2")
        ACC, PPV, NPV, recall, TNR, f1, mcc, TP, FP, TN, FN = f.printer(y_test2, svm_y2_p)
        data2_LG[feat_nums[d]] = f.pd_updater(ACC, PPV, NPV, recall, TNR, f1, mcc, TP, FP, TN, FN,
                                              "Fold-{}".format(str(fold + 1)), data1_svm[feat_nums[d]])

        print("LogisticRegression & data3")
        ACC, PPV, NPV, recall, TNR, f1, mcc, TP, FP, TN, FN = f.printer(y_test3, svm_y3_p)
        data3_LG[feat_nums[d]] = f.pd_updater(ACC, PPV, NPV, recall, TNR, f1, mcc, TP, FP, TN, FN,
                                              "Fold-{}".format(str(fold + 1)), data1_svm[feat_nums[d]])

#mean and var
for i in feat_nums:
    for columns in data1_svm[50].columns:
        data1_svm[i][columns]['mean'] = data1_svm[i][columns].mean()
        data2_svm[i][columns]['mean'] = data2_svm[i][columns].mean()
        data3_svm[i][columns]['mean'] = data3_svm[i][columns].mean()
        data1_LG[i][columns]['mean'] = data1_LG[i][columns].mean()
        data2_LG[i][columns]['mean'] = data2_LG[i][columns].mean()
        data3_LG[i][columns]['mean'] = data3_LG[i][columns].mean()

        data1_svm[i][columns]['var'] = data1_svm[i][columns].var()
        data2_svm[i][columns]['var'] = data2_svm[i][columns].var()
        data3_svm[i][columns]['var'] = data3_svm[i][columns].var()
        data1_LG[i][columns]['var'] = data1_LG[i][columns].var()
        data2_LG[i][columns]['var'] = data2_LG[i][columns].var()
        data3_LG[i][columns]['var'] = data3_LG[i][columns].var()


for i in feat_nums:
    data1_svm[i].to_csv("data1_svm_{}.csv".format(i))
    data2_svm[i].to_csv("data2_svm_{}.csv".format(i))
    data3_svm[i].to_csv("data3_svm_{}.csv".format(i))
    data1_LG[i].to_csv("data1_LG_{}.csv".format(i))
    data2_LG[i].to_csv("data2_LG_{}.csv".format(i))
    data3_LG[i].to_csv("data3_LG_{}.csv".format(i))