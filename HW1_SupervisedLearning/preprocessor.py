from sklearn.preprocessing import StandardScaler 



def normalizer(data):

    scaler = StandardScaler()
    normalized_data = scaler.fit_trainsform(data)


    return normalized_data 


