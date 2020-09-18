
import os
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import time 





def rebalance_data(df, target, strategy="SMOTE"):
    
    X = df.loc[:, df.columns != target]
    y = df.loc[:, df.columns == target]

    if len(set(y[target])) == 1:

        print ('Only one class {} of label is generated'.format(list(set(y)))[0])

        return df 
    else:

        if strategy=='SMOTE':
            sm = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=42)
            X_smote, y_smote = sm.fit_resample(X, y)
            X_smote[target] = y_smote
            df = X_smote
            #  print ('after smote rebalanced, data shape is {}'.format(df.target.value_counts()))
        if strategy=='Subsampling':

            #rus = RandomUnderSampler(sampling_strategy=0.5)
            rus = RandomUnderSampler()
            x_rus, y_rus = rus.fit_sample(X,y)
            x_rus[target] = y_rus
            df = x_rus
    return df

def fit_model(model, model_args, X_train, y_train):
    classifier = model(**model_args)
    classifier.fit(X_train, y_train)

    return classifier 


def model_performance(model, X, y, average='macro'):

    predictions = model.predict(X)
    probs = model.predict_proba(X)
    f1= f1_score(y, predictions, average=average)
    precision = precision_score(y, predictions, average=average)
    recall = recall_score(y, predictions, average=average)
    accuracy = accuracy_score(y, predictions)
    roc_auc = roc_auc_score(y, probs, multi_class='ovr', average=average)
    report = classification_report(y, predictions)

    return f1, precision, recall, accuracy, roc_auc, report




def cm_cv(m, n, X, y, score):
    cm = model_selection.cross_val_score(m, X, y, scoring=score, cv=n)


def hyperparameter_sweep(model,  X, y, k, score_fn, search_args, strategy='gridsearch'):

    if strategy == 'gridsearch':
        
        clf = GridSearchCV(model(), cv=k,  scoring = score_fn , param_grid=search_args)

    elif strategy == 'randomsearch':
        clf = RandomizedSearchCV(model(), cv=k, scoring= score_fn, param_distributions=search_args )

    search = clf.fit(X, y)
    results = search.cv_results_
    best_model = search.best_estimator_
    best_param = search.best_params_
    best_score = search.best_score_

    return {'results': results,
            'best_model': best_model,
            'best_param': best_param,
            'best_score': best_score}



def train_eval(df, model, target, config, k, balancer=False, hypertune=False, test_size = 0.3, model_args=None):
    
    if balancer:
    
        df = rebalance_data(df, target)
    
    X = df.loc[:, df.columns != target]
    y = df.loc[:, df.columns == target]
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = seed)
    
    print (X_train.shape)

    if hypertune:
        print (config[model])
        search_args = config[model]['param_values']
        model_fn =  config[model]['model']
        score_fn = config[model]['score_fn']
        start = time.time()
        results_dict = hyperparameter_sweep(model_fn, X_train, y_train, k, score_fn, search_args, strategy='gridsearch')
        classifier = results_dict['best_model']
        end = time.time()
        elapsed = end - start
        
    else:
        model_fn = config[model]['model']
        print (model_fn)
        model_args = config[model]['default_model_args']
        start = time.time()
        classifier = fit_model(model_fn, model_args, X_train, y_train)
        end = time.time()
        elapsed = end - start
        
    
        
    print ('fit model finished')   
        
    f1, precision, recall, accuracy, roc_auc, report = model_performance(classifier, X_test, y_test, average='macro')
        
    return {'classifier': classifier,
            'train_time_elapsed': elapsed,
        'f1': f1, 
            'precision': precision,
           'recall': recall,
           'accuracy': accuracy,
           'roc_auc': roc_auc, 
           'classification_report': report}



    
model_config = {
    'DecisionTreeClassifier': 
        {
            'model': DecisionTreeClassifier,
            'normalizer': False,
            'param_values':  
                {
                    "criterion": ['gini', 'entropy'],
                    "max_depth": [5, 10, 20,50,100],
                    "min_samples_leaf": [ 5, 10,20]
                },              
            'score_fn': 'f1_macro',
            'default_model_args': 
                {
                    'criterion': 'entropy',
                    'max_depth': 20,
                    'min_samples_leaf': 5
                }
        },
    'GradientBoostingClassifier':
        {
            'model': GradientBoostingClassifier,
             'normalizer': False,
            'param_values':
                {
                    "learning_rate": [0.05, 0.1, 0.5],
                    "n_estimators": [50, 100, 200],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [5, 10, 30, 50]
                },
            'score_fn': 'f1_macro',
            'default_model_args':
                {
                    "n_estimators": 200,
                    "min_samples_leaf": 50
                }
        },

    'SVM':
        {
            'model': SVC,
            'normalizer': True,
            'param_values':
                {
                    "kernel" = ['linear', ' ']
                }
        }
    
    
    
    
    
    }

