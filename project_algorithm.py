import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve,roc_auc_score
import numpy as np
import warnings
import matplotlib.pyplot as plt
import csv
import sys
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chi2_contingency
import os
import glob
import math
import seaborn as sn
import warnings 


warnings.filterwarnings('ignore')

def draw_histogram(df,attrName,binsno=3, filename=None,visualize=False):


    fig = plt.figure(figsize=[12.8, 18.8],dpi=300)
    df[attrName].value_counts().plot(kind='bar')

    if(not(filename is None)):
        fig.savefig(filename+ ".jpg")
    if (visualize):
        plt.show()
    plt.cla()
    plt.close()
def train_model(x_train, y_train,x_test,outTargetName, model_name="RandomForest"):
    if(model_name=="SVM"):
        # Create a svm Classifier
        clf = svm.SVC(kernel='linear', probability=True)  # Linear Kernel

        # Train the model using the training sets
        clf.fit(x_train, y_train[outTargetName])

        # Predict the response for test dataset
        y_pred = clf.predict(x_test)
        y_pred_prob = clf.predict_proba(x_test)
        return y_pred,y_pred_prob

    else:
        # Create the model with 100 trees
        model = RandomForestClassifier(n_estimators=100,
                                       bootstrap = True,
                                       max_features = 'sqrt')
        # Fit on training data
        # print("y_train[columns[0]]=",y_train[columns[0]])
        model.fit(x_train, y_train[outTargetName])

        # Actual class predictions
        rf_predictions = model.predict(x_test)
        rf_probs = model.predict_proba(x_test)
        return rf_predictions,rf_probs

def validate_model(y_test,rf_predictions,rf_probs,outTargetName,num_classes=3):
    # Calculate roc auc
    print(f"{outTargetName} measurement:")
    print("rf_probs=",rf_probs)
    if(num_classes>2):
        roc_auc_value = roc_auc_score(y_test[outTargetName], rf_probs, multi_class="ovo")
    else:
        roc_auc_value = roc_auc_score(y_test[outTargetName], rf_probs[:,1])
    print("roc_auc_value=",roc_auc_value)

    conf_matrix=confusion_matrix(y_test[outTargetName], rf_predictions)
    print("conf_matrix=",conf_matrix)
    precision,recall,fscore,_=precision_recall_fscore_support(y_test[outTargetName], rf_predictions, average='macro')
    print("precisin=",precision)
    print("recall=", recall)
    print("fscore=", fscore)

def main():
    df=pd.read_csv("preprocess_dataset/dataset.csv")
    X=df.iloc[:,:-3]
    Y=df.iloc[:,-3:]
    columns=list(Y.columns)
    # draw_histogram(df, columns[2],visualize=True,filename="out_figures/drivingStyle_hist")
    # print("input variable")
    # print(X)
    # print(X.columns)
    # print("output variable")
    # print(Y)
    # print(Y.columns)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=90)
    print("col[2]=",columns[2])
    rf_predictions,rf_probs=train_model(x_train, y_train, x_test, columns[2],model_name="SVM")
    validate_model(y_test, rf_predictions, rf_probs, columns[2],num_classes=2)


    # false_positive_rate, true_positive_rate, threshold = roc_curve(y_test[columns[0]], rf_predictions)
    # print("false_positive_rate=",false_positive_rate)
    # print("true_positive_rate=", true_positive_rate)
    # print("threshold=", threshold)


if __name__=='__main__':
    main()