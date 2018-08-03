import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def clusterKMeans(ndf,n):
    clusters = KMeans(n_clusters=n,init='k-means++',tol=1e-06)

    df = pd.read_csv(ndf, encoding = "ISO-8859-1")

    reduced_data = PCA(n_components=2).fit_transform(df)
    reduced_data = normalize(reduced_data,norm='l2',axis=1,copy=True,return_norm=False)
    k = clusters.fit_predict(reduced_data)

    plt.scatter(reduced_data[k == 0, 0], reduced_data[k == 0, 1], s=50, c='lightgreen', edgecolor='black',marker='o', label='cluster 1')
    #plt.scatter(reduced_data[k == 1, 0], reduced_data[k == 1, 1], s=50, c='orange', edgecolor='black', marker='v',label='cluster 2')
    #plt.scatter(reduced_data[k==2,0],reduced_data[k==2,1],s=50,c='blue',edgecolor='black',marker='s',label='cluster 3')
    plt.scatter(clusters.cluster_centers_[:, 0], clusters.cluster_centers_[:, 1], s=80, c='red', marker='*',label='centroides')

    plt.legend()
    plt.grid()
    plt.show()
clusterKMeans("2017_EM.csv",3)