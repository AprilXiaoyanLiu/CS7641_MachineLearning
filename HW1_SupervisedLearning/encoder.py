from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def feature_encode_categorical(df, categorical_columns):
    
    encoders = {}
    for column in categorical_columns:
        encoders[column] = LabelEncoder()
        print (column)
        df[column] = encoders[column].fit_transform(df[column])
    return df 


def onehot_encoder(df, categorical_columns):
    #feature encoding 
    X = feature_encode_categorical(df, categorical_columns)[categorical_columns]
    print (X)
    ohe = OneHotEncoder(sparse=False)
    X_ohe = ohe.fit_transform(X)
    
    columns = ohe.get_feature_names(categorical_columns)
    
    X_cat = pd.DataFrame(X_ohe, columns = columns)
    data = pd.concat([df, X_cat], axis=1)

    return data

def get_dummies(df, categorical_columns):
    
    return pd.get_dummies(df, columns = categorical_columns)


def target_encoder(df, categorical_columns, target, smooth=False, m=None):

    y_onehot = get_dummies(df[target], [target])
    
    
    class_names = y_onehot.columns.tolist()
    
    df = pd.concat([df, y_onehot], axis=1)
    

    for column in categorical_columns:
        for class_ in class_names:
            mean_value = _target_encoder_by_column(df, column, class_)
            df[column + '_' + str(class_)] = df[column].map(mean_value)
            
    return df.drop(categorical_columns + class_names, axis=1)

def _target_encoder_by_column(df, column, target, smooth=False, m=None):

    # modified on top of https://maxhalford.github.io/blog/target-encoding/

    if smooth:
        # global_mean 
        global_mean = df[target].mean()

        
        agg =  df.groupby(column)[target].agg(['count', 'mean'])

        mean_val = agg['mean']
        counts = agg['count']

        smooth = (counts * means + m * counts * mean) / (counts + m * counts)

        return smooth

    else:
        
        mean_val = df.groupby(column)[target].mean()
        return mean_val







    


