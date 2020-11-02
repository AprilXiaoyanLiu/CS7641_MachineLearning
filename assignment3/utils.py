
import numpy as np
import pandas as pd


#################### choosing number of k for clustering ################################
def elbow_method(k_list, X, random_state_list, save=False, filename=None, **kmeans_kwargs):
    """
    choose number of K for k-means clustering using Elbow Method
    e.g.
    kmeans_kwargs = {
    "init": "k-means++",
      "n_init": 10,
        "max_iter": 300,
      
   }
   sse_smart = elbow_method(k_list = range(10, 100, 10), X=X_train_trans, random_state_list=[0, 42, 100], **kmeans_kwargs)
   """

    # A list holds the SSE values for each k
    sse_smart = []
    
    for k in k_list:
        total = 0
        for r in random_state_list:
            kmeans = KMeans(n_clusters=k, random_state =r, **kmeans_kwargs)
            start = time.time()
            kmeans.fit(X)
            end = time.time()
            score = kmeans.inertia_
            print (k, r, score, end-start)
            
            total += score
            
        sse_smart.append(total/len(random_state_list))
        
    plt.style.use("fivethirtyeight")
    plt.plot(k_list, sse_smart, ls='--',marker='D',  markerfacecolor="r")
    plt.xticks(k_list)
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()
    if save:
        plt.savefig(filename)
        
    return sse_smart



def sihouette(k_list, X, random_state_list, save=False, filename=None, **kmeans_kwargs):
    """
    choose number of K for k-means clustering using Sihouette Coefficients
    e.g.
    kmeans_kwargs = {
    "init": "k-means++",
      "n_init": 10,
        "max_iter": 300
      
   }

    sihouette_smart_10 = sihouette(range(2,11), X_train_trans, random_state_list,  **kmeans_kwargs)
    """
    silhouette_coefficients_smart = []

    # Notice you start at 2 clusters for silhouette coefficient
    for k in k_list:
        total = 0
        for r in  random_state_list:
            kmeans = KMeans(n_clusters=k, random_state = r, **kmeans_kwargs)
            start = time.time()
            kmeans.fit(X)
            end = time.time()
            score = silhouette_score(X, kmeans.labels_, sample_size = len(df)//10)
            total += score
            print (k, r, score, end-start)
        silhouette_coefficients_smart.append(total/len(random_state_list))
    
    plt.style.use("fivethirtyeight")
    plt.plot(k_list, silhouette_coefficients_smart,ls='--',marker='D',  markerfacecolor="r" )
    plt.xticks(k_list)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()
    return silhouette_coefficients_smart


def choose_n_components_GMM(n_components, X, random_state_list, save=False, filename=None, **gmm_kwargs):
    """
    choose number of components in Gaussain Mixture Model with AIC and BIC as metrics
    e.g. 
    gmm_kwargs = {
    "init_params": "kmeans",
      "covariance_type": "full"
      
   }
    aic_score, bic_score = choose_n_components_GMM(n_components = range(10, 100, 10), X=X_train_trans, random_state_list=[0, 42, 100], **gmm_kwargs)
    """
    aic_score = []
    bic_score = []
    for n in n_components:
        total_aic = 0 
        total_bic = 0
        for r in random_state_list:
            em = GaussianMixture(n, random_state =r, **gmm_kwargs)
            start = time.time()
            em.fit(X)
            end = time.time()
            score_aic = em.aic(X)
            score_bic = em.bic(X)
            total_aic += score_aic
            total_bic += score_bic
            print (n, r, total_aic, total_bic, end-start)
        aic_score.append(total_aic/len(random_state_list))
        bic_score.append(total_bic/len(random_state_list))
        

    plt.plot(n_components, bic_score, label='BIC',ls='--',marker='D')
    plt.plot(n_components, aic_score, label='AIC',ls='--',marker='D')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    return aic_score, bic_score


######################## choose number of components for DR ###################################

def choose_component_for_PCA(X):
    pca = PCA()
    pca.fit(X)
    variance = pca.explained_variance_ratio_
    plt.figure(figsize= ( 10, 8))
    plt.plot(range(1, 109), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
    plt.title('Explained Variance by Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    return plt



from scipy.stats import kurtosis
def choose_n_ICA(n_components, X):
    
    kurtosis_scores = []
    avg_kurtosis = [np.mean(kurtosis(FastICA(n_components=n).fit_transform(X)))
              for n in n_components]

    plt.plot(n_components, avg_kurtosis)
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.ylabel('avg kurtosis')
    plt.title('N_components vs Avg Kurtosis')
    return avg_kurtosis




def get_mse_rsa(n, X, r):
    transformer = random_projection.GaussianRandomProjection(n_components=n, random_state=r)
    start = time.time()
    X_new = transformer.fit_transform(X)
    end = time.time()
    
    inverse_data = np.linalg.pinv(transformer.components_.T)
    reconstructed_data = X_new.dot(inverse_data)
    mse = mean_squared_error(reconstructed_data, X)
    return transformer, X_new, mse, end-start

def choose_n_components_rac(n_components, X, random_state_list, save=False, filename=None):
    recons_error_list = []
    for n in components:
        total = 0
        for r in random_state_list:
            _, _, mse, elapsed = get_mse_rsa(n, X, r)
            
            total += mse 
            
            print (n, r, mse, elapsed)
            
        recons_error_list.append(total/len(random_state_list))
        
    
    plt.plot(n_components, recons_error_list)
    plt.legend(loc='best')
    plt.xlabel('n_components') 
    plt.ylabel('reconstruction error')
    
    return recons_error_list




############################# PLOT Clustering Results #########################################
import seaborn as sns
rndperm = np.random.permutation(df.shape[0])

import seaborn as sns
rndperm = np.random.permutation(df.shape[0])


def plot_kmeans_cluster(n_cluster, Xtrain, Xtest, seed, init, name):
    kmeans = kmeans_clustering(n_cluster, Xtrain, seed, init)
    X_train[name] = kmeans.labels_
    X_test[name] = kmeans.predict(Xtest)
    
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue=name,
        palette=sns.color_palette("hls", n_cluster),
        data=X_train.loc[rndperm,:],
        legend="full",
        alpha=0.3
    )
    
    return kmeans



def plot_GMM_cluster(n_components, Xtrain, Xtest, random_state, name, save=False, filename=None, plot=False, **gmm_kwargs):
    
    em_model = GaussianMixture(n_components=n_components, random_state=random_state, **gmm_kwargs )
    em_model.fit(Xtrain)
    if plot:
        X_train[name] = em_model.predict(Xtrain)
        X_test[name] = em_model.predict(Xtest)
        plt.figure(figsize=(16,10))
        sns.scatterplot(
            x="pca-one", y="pca-two",
            hue=name,
            palette=sns.color_palette("hls", len(n_components)),
            data=X_train.loc[rndperm,:],
           # data = X_train[X_train[name].isin(em_model.weights_.argsort()[::-1][:5])],
            legend="full",
            alpha=0.3
        )
    return em_model




###########################  Transofrm Data ###################################################
def pca_input(n, X_train, X_test):
    pca = PCA(n_components=n)
    pca.fit(X_train)
    PCA_X_train = pca.transform(X_train)
    PCA_X_test = pca.transform(X_test)
    return PCA_X_train, PCA_X_test

