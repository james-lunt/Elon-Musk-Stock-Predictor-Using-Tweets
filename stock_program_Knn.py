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
stock = df.iloc [:,2]

#Test data for geralising
df = pd.read_csv("Dataset_Test.csv", skiprows=0)
date_test =df.iloc[:,0]
sentiment_test = df.iloc [:,1]
convert_to_timestamp(date_test)
X_test=np.column_stack((date_test,sentiment_test)) 
stock_test = df.iloc [:,2]


#Calculatestock increase / decrease
def increase_decrease(stock,ypred,keys):
    same = 0
    for j in keys:
        if((stock[j+1] - stock[j] < 0 and ypred[j+1] - ypred[j] < 0) or (stock[j+1] - stock[j] > 0 and ypred[j+1] - ypred[j] > 0)):
            same+=1

    print(same/len(keys))

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


#Cross validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
mean_error=[]; std_error=[]
neighbor_range  = [1,2,3,4,5,6,7,8]
for n in neighbor_range:
    model = KNeighborsRegressor(n_neighbors=n,weights="distance").fit(X,stock)
    temp=[]
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model.fit(X[train], stock[train])
        ypred = model.predict(X[test])
        temp.append(mean_squared_error(stock[test],ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())


#plot
plt.errorbar(neighbor_range,mean_error,yerr=std_error)
plt.xlabel('NN'); plt.ylabel("Mean square error")
plt.title("KNN Hyperparameter Cross-Validation")
plt.show()


from sklearn.model_selection import train_test_split

# using the train test split function
X_tr, X_te,y_tr, y_te = train_test_split(X,stock,shuffle=True)

#Train
model = KNeighborsRegressor(n_neighbors=3,weights="distance").fit(X_tr,y_tr)

#Predict
ypred = model.predict(X)


def results(y_pred, y_targets,keys):
    plt.xlabel('Dollars ($)')
    plt.ylabel('Prediction')
    accuracies = []
    dollar_range = [0,1,2,3,4,5,6,7,8,9,10]

    for dollar in dollar_range:
        in_range = 0
        for i in keys:
            diff = y_pred[i] - y_targets[i]
            if (diff >= (dollar*-1) and diff <= dollar):
                in_range+=1

        accuracies.append(in_range/len(keys))


    plt.plot(dollar_range,accuracies)
    plt.show()

    increase_decrease(stock,ypred,keys)


results(ypred,stock,y_te.keys())
