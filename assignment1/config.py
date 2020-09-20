from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN

model_config = {
    'DecisionTreeClassifier': 
        {
            'model': DecisionTreeClassifier,
            'normalizer': False,
            'param_values':  
                {
                    "criterion": ['gini', 'entropy'],
                    "max_depth": [5,  20,50],
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
    
    'AdaBoostClassifier':
        {
            'model': AdaBoostClassifier,
            'normalizer': False,
            'param_values':
                {
                    "base_estimator":[DecisionTreeClassifier(max_depth=1),DecisionTreeClassifier(max_depth=100,min_samples_leaf=5) ],
                    "n_estimators": [50, 200],
                    "learning_rate": [0.5, 1],
                    "algorithm": ["SAMME", "SAMME.R"]
                },
            
            'score_fn': 'f1_macro',
            'default_model_args':
                {
                    "n_estimators" : 200,
                    "learning_rate" : 0.5,
                    "base_estimator": DecisionTreeClassifier(max_depth=100,min_samples_leaf=5)
                }
        }
    ,
    
    'KNNClassifier':
        {
            'model': KNN,
            'normalizer': True,
            'param_values':
                {
                    "n_neighbors": [3, 5, 10],
                    "weights": ['uniform', 'distance'],
                    "metric": ["eculidean", "manhattan", "minkowski"]
                    
                },
            'score_fn': 'f1_macro',
            "default_model_args":
                {
                    "n_neighbors": 5,
                    "weights": "uniform"
                }
        },
    
    'MLPClassifier':
        {
            'model': MLPClassifier,
            'normalizer': True,
            'param_values':
                {
                    "hidden_layer_sizes": [32, 64, 128],
                    "max_iter": [100, 300]
                },
            'score_fn': 'f1_macro',
            "default_model_args":
            {
                "hidden_layer_sizes": 64,
                "max_iter": 300,

            }
        },
    
    'GradientBoostingClassifier':
        {
            'model': GradientBoostingClassifier,
             'normalizer': False,
            'param_values':
                {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [50,  200],
                   # "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [5, 10]
                },
            'score_fn': 'f1_macro',
            'default_model_args':
                {
                    "n_estimators": 200,
                    "min_samples_leaf": 50
                }
        },

    'SVC':
        {
            'model': SVC,
            'normalizer': True,
            'param_values':
                {
                    "kernel" : ['linear', 'rbf', 'poly'],
                    "probability":[True]
                    
                },

            'score_fn': 'f1_macro',
            'default_model_args':
                {
                    "kernel": 'rbf',
                    "C": 1,
                    "probability": True
                }

        },
    
   
    
    
    
    
    }
