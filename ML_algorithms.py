from __future__ import division, print_function, unicode_literals
import pandas as pd
import datetime
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from collections import defaultdict


'''
model_selections
'''
def best_model(classifier, params, X_train, y_train):
    p = Pipeline([('scaler',StandardScaler()) ,('forest', classifier)])
    gscv = GridSearchCV(classifier, params, scoring = 'f1', cv = 5, n_jobs = -1, verbose = 1)
    gscv.fit(X_train, y_train)
    return gscv
'''
Roc curve
'''
def report_metrics(y_test, y_pred_lbl):
    '''
    Return Precision/Recall/accuracy
    INPUT: y_test: Array of true labels
       y_pred_lbl: Array of predicted labels
    OUTPUT: Return precision, recall and accuracy score values
    '''
    precision = precision_score(y_test, y_pred_lbl)
    recall = recall_score(y_test, y_pred_lbl)
    accuracy = accuracy_score(y_test, y_pred_lbl)
    return precision, recall, accuracy

def plot_roc(y_test, y_pred_prod, name):
    '''
    Using sklearn roc_curve plot roc curve
    INPUT:
    y_test: Array of true labels
    y_pred_prod: Array of probabilities of target variable
    OUTPUT:
    None
    '''
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prod)
    plt.plot(fpr, tpr, label = name)
    plt.rcParams['font.size'] = 12
    #plt.title('ROC curve for Churn Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)

'''
Cleaning our Data
'''
def get_labels(df):
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    cutoff_date = datetime.date(2014, 6, 1)
    df[df['last_trip_date'] < cutoff_date]
    df['churn'] = (df['last_trip_date'] < cutoff_date).astype(int)
    df['recent'] = (cutoff_date-df['signup_date']).apply(lambda x: pd.Timedelta(x).days)
    return df
def fill_na(df):
    impute = Imputer()
    df['filled_by_driver'] = df['avg_rating_by_driver'].isnull().astype(int)
    df['filled_of_driver'] = df['avg_rating_of_driver'].isnull().astype(int)
    df[['avg_rating_by_driver','avg_rating_of_driver']] = impute.fit_transform(df[['avg_rating_by_driver','avg_rating_of_driver']])
    df['phone_fill'] = df['phone'].isnull().astype(int)
    df = df.fillna(method='ffill')
    return df
def remove_error(df):
    df = df[(df['avg_dist']!=0) | (df['trips_in_first_30_days']==0)]
    df = pd.get_dummies(df,drop_first=True)
    return df

df = (pd.read_csv('data/churn_train.csv')
      .pipe(get_labels)
      .pipe(fill_na)
     .pipe(remove_error)
     .drop(['signup_date','last_trip_date'],axis=1))
df_test = (pd.read_csv('data/churn_test.csv')
      .pipe(get_labels)
      .pipe(fill_na)
     .pipe(remove_error)
     .drop(['signup_date','last_trip_date'],axis=1))


y = df.pop('churn').values
X = df.values

y_actual = df_test.pop('churn').values
X_actual = df_test.values

X_train, X_test, y_train, y_test = train_test_split(X,y)
'''
computation
'''
model_dict = defaultdict(list)
random_forest = dict(n_estimators=[100,200],
                    criterion = ['gini','entropy'],
                    max_features = ['sqrt','log2', None],
                    random_state = [1],
                    min_samples_leaf = [1, 2],
                    min_samples_split = [2])
gradient_boost = {'learning_rate': [0.1, 0.5, 1],
                              'max_depth': [2, 4, 6],
                              'min_samples_leaf': [5, 10],
                              'n_estimators': [200,300],
                              'random_state': [1]}
ada_params = {'n_estimators': [50,100],
                'learning_rate': [0.1, 0.2, 0.5, 1],
                'random_state': [1]}
logistic_params = {'penalty': ['l1','l2'],
                    'C':[0.05, 0.1, 0.2, 0.5,1],
                    'random_state': [1]}
model_dict['params'] = [random_forest, gradient_boost, ada_params, logistic_params]
model_dict['models'] = [RandomForestClassifier(),GradientBoostingClassifier(), AdaBoostClassifier(), LogisticRegression()]

model_list = []
model_scores = []
for model, params in zip(model_dict['models'], model_dict['params']):
    gscv = best_model(model, params, X_train, y_train)
    model_scores.append(gscv.best_score_)
    model_list.append(gscv.best_estimator_)

print (zip(model_list,model_scores))

model_dict = defaultdict(list)
labels_dict = defaultdict(list)
for models in model_list:
    models.fit(X_train, y_train)
    model_dict[models.__class__.__name__] = models.predict_proba(X_test)[:, 1]
    labels_dict[models.__class__.__name__] = models.predict(X_test)


for model, predicted in labels_dict.iteritems():

    print ("{}: {}".format(model, report_metrics(y_test, predicted)))

for model, score in model_dict.iteritems():
    plot_roc(y_test, score, model)
plt.legend()
plt.title('ROC curve for Churn Classifier')
plt.show()

y_actual = df_test.pop('churn').values
X_actual = df_test.values
final_model = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=2,
              max_features=None, max_leaf_nodes=None,
              min_impurity_split=1e-07, min_samples_leaf=5,
              min_samples_split=2, min_weight_fraction_leaf=0.0,
              n_estimators=300, presort='auto', random_state=1,
              subsample=1.0, verbose=0, warm_start=False)
final_model.fit(X,y)
final_model.score(X_actual,y_actual)
predictions = final_model.predict(X_actual)
report_metrics(y_actual, predictions)
