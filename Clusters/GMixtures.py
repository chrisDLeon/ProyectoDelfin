import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

def gaussianMixtureC(ndf):
    df = pd.read_csv(ndf, encoding="ISO-8859-1")
    reduced_data = PCA(n_components=2).fit_transform(df)
    reduced_data = normalize(reduced_data, norm='l2', axis=1, copy=True,return_norm=False)

    gmm = GaussianMixture(n_components=4, covariance_type='full', tol=1e-3)
    gmm.fit(reduced_data)
    g = gmm.predict(reduced_data)

    plt.scatter(reduced_data[g == 0, 0], reduced_data[g == 0, 1], s=50, c='lightgreen', edgecolor='black',marker='o', label='cluster 1')
    plt.scatter(reduced_data[g == 1, 0], reduced_data[g == 1, 1], s=50, c='orange', edgecolor='black', marker='v',label='cluster 2')
    plt.scatter(reduced_data[g == 2,0], reduced_data[g == 2,1],s=50,c='blue',edgecolor='black',marker='s',label='cluster 3')
    plt.scatter(reduced_data[g == 3,0], reduced_data[g == 3,1],s=50,c='green',edgecolor='black',marker='x',label='cluster 4')

    plt.legend()
    plt.grid()
    plt.show()

gaussianMixtureC("2011_F.csv")

