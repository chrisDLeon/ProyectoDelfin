import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

def clusterMeanShift(ndf):

    df = pd.read_csv(ndf, encoding="ISO-8859-1")
    bandwidth = estimate_bandwidth(df, quantile=0.3)
    clusters = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    reduced_data = PCA(n_components=2).fit_transform(df)
    reduced_data = normalize(reduced_data,norm='l2',axis=1,copy=True,return_norm=False)
    ms = clusters.fit_predict(reduced_data)

    plt.scatter(reduced_data[ms == 0, 0], reduced_data[ms == 0, 1], s=50, c='lightgreen', edgecolor='black',marker='o', label='cluster 1')
    plt.scatter(clusters.cluster_centers_[:, 0], clusters.cluster_centers_[:, 1], s=80, c='red', marker='*',label='centroides')

    plt.legend()
    plt.grid()
    plt.show()
clusterMeanShift("2017_FC.csv")