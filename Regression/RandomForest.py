import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

regr = RandomForestRegressor(n_estimators = 700, random_state = 1)
df = pd.read_csv('IncidenciasC.csv',encoding = "ISO-8859-1")

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'bootstrap': True,
 'max_depth': 70,
 'max_features': 'auto',
 'min_samples_leaf': 4,
 'min_samples_split': 10,
 'n_estimators': 400}

rf_random = RandomizedSearchCV(estimator = regr, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

train,test = train_test_split(df, test_size=0.2, random_state=1) #Subdivisión de los datos

train_X = train[['CVE_ENT','CVE_MUN','TMODVICC_2017','CVE_MOD','CVE_TIPO','CVE_SUBTIPO']] # Datos con los que se entrena
train_y=train['TOTAL_DELITOS']
test_X= test[['CVE_ENT','CVE_MUN','TMODVICC_2017','CVE_MOD','CVE_TIPO','CVE_SUBTIPO']] # Datos con los que se prueba
test_y=test['TOTAL_DELITOS']

rf_random.fit(train_X,train_y)

o = rf_random.predict(test_X)

o = (o*10).astype('int')
o = (o).T
o = o.tolist()

test_y = test_y.tolist()

hits = 0
for i in range(1, len(o)):
    if o[i] == test_y[i]:
        hits += 1

print('performance:', round((hits / len(o)) * 100), '%')