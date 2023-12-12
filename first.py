import pickle
import Funcs as f
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import numpy as np

ds1, ds2, ds3, labels, folds = f.data_reader()
print("Data preparation is done!")

Data = np.concatenate([ds1,ds2,ds3],axis = -1)

with open('data.pickle', 'rb') as handle:
    data = pickle.load(handle)[15]

Data = Data[:,data[:200]]

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