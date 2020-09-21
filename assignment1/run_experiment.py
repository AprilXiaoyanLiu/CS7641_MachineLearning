# +
import argparse
import pandas as pd
from utils.encoder import onehot_encoder, feature_encode_categorical, target_encoder
from load_data import train_test_generate
from model_train_eval import fit, evaluate
from config import model_config
import numpy as np

from visualization import plot_learning_curve



config_data = {
    'animal':
        {'path': 'data/animal_clean.csv',
         'sep': ',',
         'encoder': "OneHotEncoder",
        'features': ','.join(['AnimalType', 'SexuponOutcome',  'Breed', 'Color', 'Sex', 'neuter_status']),
        'target': 'OutcomeType'},
    
    'adult':
        {
            'path': 'data/adult.csv',
            'sep': ',',
            'encoder': "OneHotEncoder",
            "features": ','.join(['workclass',
                                'education',
                                'marital-status',
                                'occupation',
                                'relationship',
                                'race',
                                'sex',
                                'native-country']),
                                 
             "target": "income"                    
            
        }
}


def fit_eval(dataset, model, visualize=True, hypertune=False):
    config1 = config_data[dataset]
    path = config1['path']
    sep = config1['sep']
    features = config1['features'].split(',')
    target = config1['target']
    data = pd.read_csv(path, sep=sep)
    
    print ('data loaded successfully')
    X_train, X_test, y_train, y_test = train_test_generate(data, target=target, features=features, encoder=config1['encoder'])
    print ('-------------------------------------------------')
    print ('start to train model {}'.format(model))
    classifier_dict = fit( X_train, y_train, X_test, y_test, model, target, model_config, k=3, hypertune=hypertune, visualize=visualize)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform Supervised Learning Experiments')
    parser.add_argument('--dataset', type=str, default='animal', help='which dataset to choose')
    parser.add_argument('--visualize', action='store_false', help="whether generate visualization or not")
    parser.add_argument('--hypertune', action='store_false', help="whether to do gridsearchcv hyperparameter tunning")
    args = parser.parse_args()
    
    finish_training = []
    for key in model_config:
        
        if key not in finish_training:
            fit_eval(args.dataset, key)
            finish_training.append(key)


            

