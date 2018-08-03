import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

df_2011 = pd.read_csv("2011_F.csv", encoding = "ISO-8859-1")
df_2012 = pd.read_csv("2012_F.csv", encoding = "ISO-8859-1")
df_2013 = pd.read_csv("2013_F.csv", encoding = "ISO-8859-1")
df_2014 = pd.read_csv("2014_F.csv", encoding = "ISO-8859-1")
df_2015 = pd.read_csv("2015_F.csv", encoding = "ISO-8859-1")
df_2016 = pd.read_csv("2016_F.csv", encoding = "ISO-8859-1")
df_2017 = pd.read_csv("2017_F.csv", encoding = "ISO-8859-1")
df_2017EM = pd.read_csv("2017_EM.csv", encoding = "ISO-8859-1")

def calculaNClusters(df):
    reduced_data = PCA(n_components=2).fit_transform(df)
    reduced_data = normalize(reduced_data,norm='l2',axis=1,copy=True,return_norm=False)
    distortions = []
    for i in range(1, 11):
         km = KMeans(n_clusters=i,
                     init='k-means++',
                     max_iter=300,
                     random_state=0)
         km.fit(reduced_data)
         distortions.append(km.inertia_)

    plt.plot(range(1,11), distortions, marker='*',c="black"  )
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()

calculaNClusters(df_2017EM)
