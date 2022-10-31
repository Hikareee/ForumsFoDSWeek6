import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import  SVC
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score

df = pd.read_csv('Seasons_Stats.csv')
df['FG% Class']=np.where(df['FG%']>df['FG%'].mean(),'Good','Bad')
df

df.info()

X=df.iloc[0:82,[33,52]] 
print(X)
y=df.loc[0:81,'FG% Class']
print(y)

svm = SVC(kernel='rbf', random_state=1, gamma=2, C=4.2)


svm.fit(X, y)


y_pred=svm.predict(X)


print(accuracy_score(y, y_pred))
