# -*- coding: utf-8 -*-
"""
Nafis Asghari
Student Number: 20196621
MMAI
Cohort: 2020
Course: MMAI 891 - Natural Language Processing
June 15, 2020


Submission to Question 2 - ML Model and Evaluation steps
"""

################################
### Import packages and Data ###
################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib.ticker as ticker
import os

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, cohen_kappa_score, log_loss, f1_score, precision_recall_curve


######################################
# ---------Machine Learning---------#
######################################
def train_model_func(x, y, estimator, param_grid, cv=5, scoring='f1'):
    grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                               scoring=scoring)
    grid_search.fit(x, y)
    
    print ("best parameters: ", grid_search.best_params_)
    #print(f"best {scoring} score during training: ", grid_search.best_score_)
    print("-------"*10)
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print(f"mean of {scoring} scores during validation: {sum(means)/len(means):0.3f}")
    print("-------"*10)
    best_model = grid_search.best_estimator_
    print ("best_model: ", best_model)

    return best_model



def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, out_dir = None):
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="lower left", fontsize=16)
    plt.ylim([0, 1])
    plt.savefig(os.path.join(out_dir , 'precision_recall_plot.png'))


def metrics(df_test, clf, x_test, y_test, return_results = False , thrsh = 0.5, out_dir = None):
    
    y_proba = clf.predict_proba(x_test)
    
    y_pred = [np.where(x[1] > thrsh , 1 , 0) for x in y_proba]
    #y_pred = clf.predict(x_test)


    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    print("Confusion Matrix:")
    print(cm)
    
    print("-----" * 5)
    print(classification_report(y_test, y_pred))

    print("-----" * 5)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy: {0:.3f}".format(acc))

    kappa = cohen_kappa_score(y_test, y_pred)
    print("Kappa: {0:.3f}".format(kappa))

    logloss = log_loss(y_test, y_proba)
    print("Log loss: {0:.3f}".format(logloss))
     
    f1 = f1_score(y_test, y_pred)
    print("f1: {0:.3f}".format(f1))

    metrics_df = pd.DataFrame({"Accuracy": [acc], "f1": [f1], "Kappa": [kappa], 'log_loss': [logloss]}).round(3)
    metrics_df.to_csv(os.path.join(out_dir, 'metrics_ml_based.csv') , index = False) 
    
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba[:, 1])
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds, out_dir)
    
    df_test['y_pred'] = y_pred
    df_test['y_proba'] = y_proba[:, 1]
    df_test.to_csv(os.path.join(out_dir, 'df_preds_ml_based.csv'))
    
    wrong_preds = df_test[df_test.Polarity != df_test.y_pred]
    
    wrong_preds.to_csv(os.path.join(out_dir, f'wrong_preds_ml_{thrsh}_test.csv'))
    
    return 

    
if __name__ == '__main__':  
    pass

    
