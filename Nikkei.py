import pandas_datareader.data as web
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


#Get Nikkei stock average from web
nikkei = web.DataReader("NIKKEI225", "fred", "1950/1/1")
plt.plot(nikkei)

#Transform to Logarithmic rate of return
nikkei['log_change']=np.log(nikkei['NIKKEI225'])-np.log(nikkei['NIKKEI225'].shift(1))
nikkei2=nikkei['log_change'].values

#Reshape for matrix with 7days data + 8th-day data as one line
term = 7
pricedata = []

length=len(nikkei2)
for i in range(0,length-term-1):
    pricedata.append(nikkei2[i:i+term+1])

df=pd.DataFrame(np.array(pricedata).reshape(-1,8))
df.columns = ['1st_day', '2nd_day', '3rd_day','4th_day', '5th_day', '6th_day','7th_day','8th_day_pred']

#Delete N/A data
df=df.dropna()
df=df.reset_index(drop=True)

#Transfrom to boolian based on the increase/decrease from 1 day before
for i in range(0,len(df)):
    if df['8th_day_pred'][i]>=0:
        df['8th_day_pred'][i]=1
    elif df['8th_day_pred'][i]<0:
        df['8th_day_pred'][i]=0

#Separate conditions and answer
x=np.array(df.drop(['8th_day_pred'],axis=1))
y=np.array(df['8th_day_pred'], dtype=np.int16)

#Separate train data and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0,shuffle=False)

#Set grid search condition
#grid_param = [{"n_estimators": [50,100],
#               "max_depth"   : [None,5],
#              }]

grid_param = [{"max_depth"   : [None]
              }]

#Set cross validation condition
kfold_cv = KFold(n_splits=5, shuffle=True)

#Set algorizm
clf = GridSearchCV(RandomForestClassifier(), grid_param, cv=kfold_cv)

#Machine learning
clf.fit(x_train, y_train)
print("Best parameter = ", clf.best_estimator_)

#Prediction
y_pred = clf.predict(x_test)

#output
print(classification_report(y_pred, y_test))
print("Accuracy [%] = " , accuracy_score(y_test, y_pred))
print(datetime.datetime.now().strftime('%Y.%m.%d. %H:%M:%S'), "  Analysis have done")