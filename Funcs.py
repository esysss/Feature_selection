from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from scipy.io import loadmat as mat
import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

def data_reader():
    # grab the data sets
    ds = mat('Features.mat')
    ds1 = ds['Feat1']
    ds2 = ds['Feat2']
    ds3 = ds['Feat3']

    # grab the labels
    ds = mat('Label_B.mat')
    labels = ds['GT']
    idx = ds['Index']
    # turn to python idx
    idx -= 1

    # the fold addresses
    ds = mat('Partition.mat')
    folds = ds['Partition']

    # use the indexes to rebuild the data sets
    ds1 = ds1[idx[:, 0], :]
    ds2 = ds2[idx[:, 0], :]
    ds3 = ds3[idx[:, 0], :]

    return ds1, ds2, ds3, labels[:,0], folds

def fsf(X, y, n_features=50):
    # knn = KNeighborsClassifier(n_neighbors=15)
    # sfs = SequentialFeatureSelector(knn, n_features_to_select=n_features)
    # sfs.fit(X, y)
    # return sfs.transform(X)
    clf = ExtraTreesClassifier(n_estimators=n_features)
    clf = clf.fit(X, y)
    model = SelectFromModel(clf, prefit=True,max_features=n_features)
    return model.transform(X)

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)


def printer(y_actual, y_predict):
    TP, FP, TN, FN = perf_measure(y_actual, y_predict)
    mcc = matthews_corrcoef(y_actual, y_predict)
    f1 = f1_score(y_actual, y_predict)
    recall = recall_score(y_actual, y_predict)
    ACC = (TP + TN) / (TP + FP + FN + TN)

    if TP != 0:
        PPV = TP / (TP + FP)
    else:
        PPV = 0

    if TN != 0:
        NPV = TN / (TN + FN)
    else:
        NPV = 0

    if TN !=0:
        TNR = TN / (TN + FP)
    else:
        TNR = 0

    return ACC, PPV, NPV, recall, TNR, f1, mcc, TP, FP, TN, FN

def pd_creator():
    df = pd.DataFrame(columns=["Acc", "Prec_Class1", "Prec_Class0", "Recall", "TNR", "F1", "MCC", "TP", "FP", "FN", "TN"],
                 index=["Fold-1", "Fold-2", "Fold-3", "Fold-4", "Fold-5",
                        "Fold-6", "Fold-7", "Fold-8", "Fold-9", "Fold-10", "mean", "var"])

    return df

def pd_updater(ACC, PPV, NPV, recall, TNR, f1, mcc, TP, FP, TN, FN, idx, df):
    df['Acc'][idx] = ACC
    df['Prec_Class1'][idx] = PPV
    df['Prec_Class0'][idx] = NPV
    df['Recall'][idx] = recall
    df['TNR'][idx] = TNR
    df['F1'][idx] = f1
    df['MCC'][idx] = mcc
    df['TP'][idx] = TP
    df['FP'][idx] = FP
    df['FN'][idx] = FN
    df['TN'][idx] = TN

    return df