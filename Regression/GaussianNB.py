import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

nb = GaussianNB()
df = pd.read_csv('IncidenciasC.csv',encoding = "ISO-8859-1")

train,test = train_test_split(df, test_size=0.2, random_state=1)

train_X = train[['CVE_ENT','CVE_MUN','TMODVICC_2017','CVE_MOD','CVE_TIPO','CVE_SUBTIPO','ENERO','FEBRERO','MARZO','ABRIL','MAYO','JUNIO','JULIO','AGOSTO','SEPTIEMBRE','OCTUBRE','NOVIEMBRE','DICIEMBRE']]
train_y=train['TOTAL_DELITOS']
test_X= test[['CVE_ENT','CVE_MUN','TMODVICC_2017','CVE_MOD','CVE_TIPO','CVE_SUBTIPO','ENERO','FEBRERO','MARZO','ABRIL','MAYO','JUNIO','JULIO','AGOSTO','SEPTIEMBRE','OCTUBRE','NOVIEMBRE','DICIEMBRE']]
test_y=test['TOTAL_DELITOS']


nb.fit(train_X,train_y)

o = nb.predict(test_X)

o = (o*10).astype('int')
o = (o).T
o = o.tolist()

test_y = test_y.tolist()
hits = 0

for i in range(1, len(o)):
    if o[i] == test_y[i]:
        hits += 1

print('performance:', round((hits / len(o)) * 100), '%')



