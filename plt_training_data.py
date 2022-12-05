import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("Dataset10Quantines.csv", skiprows=0)
date =df.iloc[:,0]
sentiment = df.iloc [:,1]
#Convert date to float timestamp
for i in range(len(date)):
    date[i] = time.mktime(time.strptime(date[i], '%Y-%m-%d'))


X=np.column_stack((date,sentiment)) 
stock = df.iloc [:,2]

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X[:,0],X[:,1],stock)
ax.set_xlabel('Date')
ax.set_ylabel('Sentiment')
ax.set_zlabel('Stock')
plt.show()
