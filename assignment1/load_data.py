

from sklearn.model_selection import train_test_split
from utils.encoder import onehot_encoder, feature_encode_categorical, target_encoder



def train_test_generate(df, target,  encoder=None, features=None, test_size=0.3):
    
    if not encoder:
        if encoder not in ['OneHotEncoder', 'LabelEncoder', 'TargetEncoder']:
            raise ValueError('encoder must be None or OneHotEncoder or LabelEncoder or TargetEncoder !')
            
        if not features:
            raise ValueError('if encoder is not true, features need to be specified')
    
    if encoder == "OneHotEncoder":
        df = onehot_encoder(df, features)
        
    elif encoder == "LabelEncoder" or encoder == "TargetEncoder":
        df = feature_encode_categorical(df, features)
        
    
    X = df.loc[:, df.columns != target]
    y = df.loc[:, df.columns == target]
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
    
    if encoder == 'TargetEncoder':
        X_train, X_test = target_encoder(X_train, y_train, X_test, features, target)
    
    
    
    return X_train, X_test, y_train, y_test
