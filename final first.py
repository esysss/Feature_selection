import Funcs as f
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

ds1, ds2, ds3, labels, folds = f.data_reader()
print("Data preparation is done!")

classifier1 = svm.LinearSVC().fit(ds1, labels)
classifier2 = svm.LinearSVC().fit(ds2, labels)
classifier3 = svm.LinearSVC().fit(ds3, labels)

coeff1 = np.abs(classifier1.coef_)[0,:]
coeff2 = np.abs(classifier2.coef_)[0,:]
coeff3 = np.abs(classifier3.coef_)[0,:]

data1 = sorted(range(len(coeff1)), key=coeff1.__getitem__,reverse=True)
data2 = sorted(range(len(coeff2)), key=coeff2.__getitem__,reverse=True)
data3 = sorted(range(len(coeff3)), key=coeff3.__getitem__,reverse=True)

Data1 = ds1[:,data1[:100]]
Data2 = ds2[:,data2[:100]]
Data3 = ds3[:,data3[:100]]

Data = np.concatenate([Data1,Data2,Data3],axis = -1)

data_svm = f.pd_creator()
data_LG = f.pd_creator()

for fold in range(10):
    print("fold = ",fold+1)
    train = np.where(folds[:,fold] == 0)[0]
    test = np.where(folds[:,fold] == 1)[0]

    x_train = Data[train, :]
    y_train = labels[train]

    x_test = Data[test, :]
    y_test = labels[test]

    svm_classifier = svm.LinearSVC().fit(x_train, y_train)
    LG_classifier = LogisticRegression().fit(x_train, y_train)

    svm_y = svm_classifier.predict(x_train)
    LG_y = LG_classifier.predict(x_train)

    ACC, PPV, NPV, recall, TNR, f1, mcc, TP, FP, TN, FN = f.printer(y_train, svm_y)
    data_svm = f.pd_updater(ACC, PPV, NPV, recall, TNR, f1, mcc, TP, FP, TN, FN,
                                           "Fold-{}".format(str(fold + 1)), data_svm)

    ACC, PPV, NPV, recall, TNR, f1, mcc, TP, FP, TN, FN = f.printer(y_train, LG_y)
    data_LG = f.pd_updater(ACC, PPV, NPV, recall, TNR, f1, mcc, TP, FP, TN, FN,
                            "Fold-{}".format(str(fold + 1)), data_LG)


#mean and var
for columns in data_svm.columns:
    data_svm[columns]['mean'] = data_svm[columns].mean()
    data_LG[columns]['mean'] = data_LG[columns].mean()

    data_svm[columns]['var'] = data_svm[columns].var()
    data_LG[columns]['var'] = data_LG[columns].var()

data_svm.to_csv("data_svm.csv")
data_LG.to_csv("data_LG.csv")