import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def feature_encode_categorical(df, categorical_columns):
    
    df[categorical_columns] = df[categorical_columns].fillna('Unknown')
    
    encoders = {}
    for column in categorical_columns:
        encoders[column] = LabelEncoder()
        #print (column)
        df[column] = encoders[column].fit_transform(df[column])
    return df 


# +

def onehot_encoder(df, categorical_columns):
    #feature encoding 
    X = feature_encode_categorical(df, categorical_columns)[categorical_columns]
   # print (X)
    ohe = OneHotEncoder(sparse=False)
    X_ohe = ohe.fit_transform(X)
   # print (X_ohe.shape)
    
    columns = ohe.get_feature_names(categorical_columns)
   # print (columns)
    
    X_cat = pd.DataFrame(X_ohe, columns = columns).reset_index().drop('index',axis=1)
    data = pd.concat([df.reset_index().drop('index',axis=1), X_cat], axis=1)

    return data.drop(categorical_columns,axis=1)


# -

def get_dummies(df, categorical_columns):
    
    return pd.get_dummies(df, columns = categorical_columns)


def target_encoder(X_train, y_train, X_test, categorical_columns, target, smooth=False, m=None):
    
    if y_train[target].nunique() > 2:
    
        X_train[target] = y_train
        y_onehot = get_dummies(X_train[target], [target])


        class_names = y_onehot.columns.tolist()

        df = pd.concat([X_train, y_onehot], axis=1)


        for column in categorical_columns:
            for class_ in class_names:
                mean_value = _target_encoder_by_column(df, column, class_)
                df[column + '_' + str(class_)] = df[column].map(mean_value)
                X_test[column+'_' + str(class_)] = X_test[column].map(mean_value)

        return df.drop(categorical_columns + class_names + [target], axis=1), X_test.drop(categorical_columns, axis=1)
    
    else:
        for column in categorical_columns:
            enc = LabelEncoder()
            target_enc = enc.fit_transform(y_train)
            
            X_train[target] = target_enc
            df = X_train
            mean_value = _target_encoder_by_column(df, column, target)
            df[column] = df[column].map(mean_value)
            X_test[column] = X_test[column].map(mean_value)
            
        return df.drop( [target], axis=1), X_test

def _target_encoder_by_column(df, column, target, smooth=False, m=None):

    # modified on top of https://maxhalford.github.io/blog/target-encoding/

    if smooth:
        # global_mean 
        if not m:
            raise ValueError ("m needs to be a float when smooth is set to be True!")
        global_mean = df[target].mean()

        
        agg =  df.groupby(column)[target].agg(['count', 'mean'])

        mean_val = agg['mean']
        counts = agg['count']

        smooth = (counts * mean_val + m * counts * global_mean) / (counts + m * counts)

        return smooth

    else:
        
        mean_val = df.groupby(column)[target].mean()
        return mean_val







    


