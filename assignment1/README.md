# Project1 - Supervised Learning

Run six classifiers on two datasets. Six classifiers include DecisionTreeClassifier, AdaboostClassifier, GradientBoostingClassifier, SVC, KNeighborsClassifier, MLPClassifier.

## Datasets

The default encoder for two datasets are OneHotEncoder. You can also select "TargetEncoder" or "LabelEncoder" to handle categorical data. 

Dataset 1: Shelter Animal Outcome. The cleaned version after feature engineer is used in this project. Path: "data/animal_clean.csv"
```json
'animal':
        {'path': 'data/animal_clean.csv',
         'sep': ',',
         'encoder': "OneHotEncoder",
        'features': ','.join(['AnimalType', 'SexuponOutcome',  'Breed', 'Color', 'Sex', 'neuter_status']),
        'target': 'OutcomeType'}
```      
Dataset 2: Adult data set. Dataset from UCI ML Repository 

```json
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
```

## Requirements 
python 3.5+ is needed 
```
pip install -r requirements.txt 
```
## How to:

To run dataset 1
```
python run_experiment.py --dataset 'animal' --visualize
```
To run dataset 2
```
python run_experiment.py --dataset 'adult' --visualize
```
