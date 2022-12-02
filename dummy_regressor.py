import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#Calculatestock increase / decrease
def increase_decrease(stock,ypred,keys):
    same = 0
    for j in keys:
        if((stock[j+1] - stock[j] < 0 and ypred[j+1] - ypred[j] < 0) or (stock[j+1] - stock[j] > 0 and ypred[j+1] - ypred[j] > 0)):
            same+=1

    print(same/len(keys))

#Training data
#Convert date to float timestamp
def convert_to_timestamp(dates):
    for i in range(len(dates)):
        dates[i] = time.mktime(time.strptime(dates[i], '%Y-%m-%d'))

df = pd.read_csv("Dataset1.csv", skiprows=0)
date =df.iloc[:,0]
sentiment = df.iloc [:,1]
convert_to_timestamp(date)
X=np.column_stack((date,sentiment)) 
y = df.iloc [:,2]

from sklearn.model_selection import train_test_split

# using the train test split function
X_train, X_test,Y_train, Y_test = train_test_split(X,y,shuffle=True)

import numpy as np
from sklearn.dummy import DummyRegressor
dummy_regr = DummyRegressor(strategy="median")
dummy_regr.fit(X_train, Y_train)
ypred = dummy_regr.predict(X)

correct =0
for i in Y_test:
    if i == ypred[0]:
        correct+=1

def results(y_pred, y_targets,keys):
    plt.xlabel('Dollar Range ($)')
    plt.ylabel('Prediction Accuracy')
    plt.title('Baselines Predictor using Median strategy')
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

    increase_decrease(y_targets,ypred,keys)
    #This MightBreak


results(ypred,y,Y_test.keys())
