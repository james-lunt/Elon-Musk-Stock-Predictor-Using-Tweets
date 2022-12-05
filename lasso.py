import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#Convert date to float timestamp
def convert_to_timestamp(dates):
    for i in range(len(dates)):
        dates[i] = time.mktime(time.strptime(dates[i], '%Y-%m-%d'))


#Training data
df = pd.read_csv("Dataset1.csv", skiprows=0)
date =df.iloc[:,0]
sentiment = df.iloc [:,1]
convert_to_timestamp(date)
X=np.column_stack((date,sentiment)) 
Y = df.iloc [:,2]


#Calculatestock increase / decrease
def increase_decrease(stock,ypred,keys):
    same = 0
    for j in keys:
        if((stock[j+1] - stock[j] < 0 and ypred[j+1] - ypred[j] < 0) or (stock[j+1] - stock[j] > 0 and ypred[j+1] - ypred[j] > 0)):
            same+=1

    print(same/len(keys))


#Cross validate
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


c = [0.001,0.01,0.1,1]
q_range = [2,3,4,5,8,10]
bms = []
for q in q_range:
    X_p = PolynomialFeatures(q).fit_transform(X)
    mean=[] 
    std=[]
    for i in c:
        model = Lasso(alpha=(1/(2*i)))
        temp = []
        kf = KFold(n_splits = 5)
        for train, test in kf.split(X_p):
            model.fit(X_p[train],Y[train])
            pred = model.predict(X_p[test])
            temp.append(mean_squared_error(Y[test],pred))
        mean.append(np.array(temp).mean())
        std.append(np.array(temp).std())
    bms.append(min(mean))

    plt.errorbar(c,mean,yerr=std)
    plt.xlabel('C')
    plt.ylabel('Mean square error')
    plt.title("Prediction error vs C")
    plt.show()

bms1 = min(bms)
print(bms1)
