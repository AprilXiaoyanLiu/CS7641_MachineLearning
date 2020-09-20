
import os
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler 

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import time 
import pickle
from visualization import plot_learning_curve




def rebalance_data(X, y,  strategy="SMOTE"):
    
   # X = df.loc[:, df.columns != target]
   # y = df.loc[:, df.columns == target]

    
    try:
        if strategy=='SMOTE':
            sm = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=42)
            print ('this should happen')
            X_sample, y_sample = sm.fit_resample(X, y)
            #X_smote[target] = y_smote
            #df = X_smote
            #print ('after smote rebalanced, data shape is {}'.format(df.target.value_counts()))
        elif strategy=='Subsampling':

            #rus = RandomUnderSampler(sampling_strategy=0.5)
            rus = RandomUnderSampler()
            X_sample, y_sample = rus.fit_sample(X,y)
           # x_rus[target] = y_rus
           # df = x_rus
        else:
            raise ValueError('strategy is not supported!')
        
        return X_sample, y_sample
    
    except:
        return X,y

def fit_model(model, model_args, X_train, y_train):
    classifier = model(**model_args)
    classifier.fit(X_train, y_train)

    return classifier 


def evaluate(model, X, y, average='macro', save=True):

    predictions = model.predict(X)
    probs = model.predict_proba(X)
    f1= f1_score(y, predictions, average=average)
    precision = precision_score(y, predictions, average=average)
    recall = recall_score(y, predictions, average=average)
    accuracy = accuracy_score(y, predictions)
    enc = LabelEncoder()
    
    if probs.shape[1] > 2:
    
        roc_auc = roc_auc_score(y.iloc[:,0], probs, multi_class='ovr', average=average)
    else:
        roc_auc = roc_auc_score(y.iloc[:,0], probs[:,1], multi_class='ovr', average=average)
    report = classification_report(y, predictions)
    
    result = {
            'f1': f1, 
            'precision': precision,
           'recall': recall,
           'accuracy': accuracy,
           'roc_auc': roc_auc, 
           'classification_report': report
            }
    
    print ("roc_auc is {}".format(roc_auc))
    print (report)
    
    if save:
        
        with open('results/{}.pickle'.format(model), 'wb') as handle:
            pickle.dump(result, handle)
        

    return result





def normalizer(X_train, X_test ):

    scaler = StandardScaler()
    X_train_trans = scaler.fit_transform(X_train)
    X_test_trans = scaler.transform(X_test)


    return X_train_trans, X_test_trans 




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


# +
def fit(X_train, y_train, X_test, y_test, model, target, config, k, balancer=True, hypertune=False, test_size = 0.3, model_args=None, save=True, visualize=True):
    
    #X = df.loc[:, df.columns != target]
    #y = df.loc[:, df.columns == target]
    

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
    
    if balancer:

        
        X_train, y_train = rebalance_data(X_train, y_train)
        
    
    if config[model]['normalizer']:
        print ('normalize data first!')
        X_train, X_test = normalizer(X_train, X_test)
        
    
    #print (X_train.shape)

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
        info = results_dict['results']
        
    else:
        model_fn = config[model]['model']
        #print (model_fn)
        if not model_args:
            model_args = config[model]['default_model_args']
        start = time.time()
        print ('start')
        classifier = fit_model(model_fn, model_args, X_train, y_train)
        print ('end')
        end = time.time()
        elapsed = end - start
        info = None
        
    print ('fit model finished')  
    
    result = {'classifier': classifier,
            'train_time_elapsed': elapsed,
           'info': info }
    
    if save:
        with open('model/{}.pickle'.format(model), 'wb') as handle:
            pickle.dump(result, handle)
            
            
    performance_result = evaluate(classifier, X_test, y_test, average='macro', save=True)
    
    if visualize:
        
        plot_learning_curve(classifier, X_train, y_train, ylim=(0.4, 1.01), cv=3, n_jobs=4, train_sizes=np.linspace(0.1, 1.0, 5))
    
        
    print ('plot finished')
    
        
    return {**result, **performance_result}
#{
#        'classifier': classifier,
#            'train_time_elapsed': elapsed,
#           'info': info }
        
    
        
    
# -

def fit_eval(X_train, y_train, X_test, y_test, model, target, config, k, balancer=True, hypertune=False, test_size = 0.3, model_args=None, save=True):
    
    #X = df.loc[:, df.columns != target]
    #y = df.loc[:, df.columns == target]
    

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
    
    fit()
        
    
        
    print ('fit model finished')   
        
    f1, precision, recall, accuracy,report = model_performance(classifier, X_test, y_test, average='macro')
        
    return {'classifier': classifier,
            'train_time_elapsed': elapsed,
            'f1': f1, 
            'precision': precision,
           'recall': recall,
           'accuracy': accuracy,
           'roc_auc': roc_auc, 
           'classification_report': report,
           'info': info }

"""
def train_eval(X_train, y_train, X_test, y_test, model, target, config, k, balancer=True, hypertune=False, test_size = 0.3, model_args=None):
    
    #X = df.loc[:, df.columns != target]
    #y = df.loc[:, df.columns == target]
    

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
    
    if balancer:

        
        X_train, y_train = rebalance_data(X_train, y_train)
        
    
    if config[model]['normalizer']:
        print ('normalize data first!')
        X_train, X_test = normalizer(X_train, X_test)
        
    
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
        info = results_dict['results']
        
    else:
        model_fn = config[model]['model']
        print (model_fn)
        if not model_args:
            model_args = config[model]['default_model_args']
        start = time.time()
        print ('start')
        classifier = fit_model(model_fn, model_args, X_train, y_train)
        print ('end')
        end = time.time()
        elapsed = end - start
        info = None
        
    
        
    print ('fit model finished')   
        
    f1, precision, recall, accuracy,report = model_performance(classifier, X_test, y_test, average='macro')
        
    return {'classifier': classifier,
            'train_time_elapsed': elapsed,
            'f1': f1, 
            'precision': precision,
           'recall': recall,
           'accuracy': accuracy,
           'roc_auc': roc_auc, 
           'classification_report': report,
           'info': info }
"""












