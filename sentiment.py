import numpy as np
import pandas as pd
import seaborn as sns
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

##### Reading in the Data #####
df = pd.read_csv(r"C:\Users\Niall Groves\Desktop\MachineLA\GroupAssignment\elonTestData.csv")
x = df.iloc [:,0]
y = df.iloc [:,1]

##### Getting Median of the Likes #####
median = np.median(y)
#print("Median: ", median)

##### Assigning if Tweet is Positive or Negative Based on the Mean #####
i = 0
for num in y:
    if y[i] >= median:
        y[i] = 1
    else:
        y[i] = 0
    i = i + 1
i = 0

##### Getting Rid of Stop Words #####
##### Creating a Bag of Words #####
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words='english')          #brought down to 527 from 638
X_train_counts = count_vect.fit_transform(x)
#print(X_train_counts)

##### Tones down the weight applied to very frequent words #####
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
#print(X_train_tf)

##### Splitting up the Data #####
#from sklearn.model_selection import train_test_split
#X_train_tf, xtest, y, ytest = train_test_split(X_train_tf, y, test_size=0.2)
#####a ^^^^ dont need in kfold i dont think

##### Training a Model #####
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from numpy import mean
from numpy import std
from matplotlib import pyplot

penalty = ['none', 'l1', 'l2', 'elasticnet']
weights = (0.01, 0.1, 1, 10)
repeats = range(5,10)
best_mean = 100
results = list()
#for loop to loop through penalyt choice
#for loop to cycle through differnt penalyt weights
#for loop to cycle through kfold
model = LogisticRegression()
for r in repeats:
    cv = RepeatedKFold(n_splits=5, n_repeats=r, random_state=1)
    scores = cross_val_score(model, X_train_tf, y, scoring='accuracy', cv=cv, n_jobs=-1)
    #print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    if(mean(scores) <= best_mean):
        best_mean = mean(scores)
    results.append(scores)
results.clear()
print("Penalty: None")
print("Best mean: ", best_mean)
print()

"""
best_mean = 100
for c in weights:
    model = LogisticRegression(penalty = penalty[1], C = c, solver = 'liblinear')
    for r in repeats:
        cv = RepeatedKFold(n_splits=5, n_repeats=r, random_state=1)
        scores = cross_val_score(model, X_train_tf, y, scoring='accuracy', cv=cv, n_jobs=-1)
        #print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
        if(mean(scores) <= best_mean):
            best_mean = mean(scores)
            best_weight = c
        results.append(scores)
    results.clear()
print("Penalty: l1")
print("Best mean: ", best_mean)
print("Best weight: ", best_weight) 
print()
"""
best_mean = 100
for c in weights:
    model = LogisticRegression(penalty = penalty[2], C = c)
    for r in repeats:
        cv = RepeatedKFold(n_splits=5, n_repeats=r, random_state=1)
        scores = cross_val_score(model, X_train_tf, y, scoring='accuracy', cv=cv, n_jobs=-1)
        #print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
        if(mean(scores) <= best_mean):
            best_mean = mean(scores)
            best_weight = c
        results.append(scores)
    results.clear()
print("Penalty: l2")
print("Best mean: ", best_mean)
print("Best weight: ", best_weight) 
print()

""""
best_mean = 100
for c in weights:
    model = LogisticRegression(penalty = penalty[3], C = c, solver = 'saga')
    for r in repeats:
        cv = RepeatedKFold(n_splits=5, n_repeats=r, random_state=1)
        scores = cross_val_score(model, X_train_tf, y, scoring='accuracy', cv=cv, n_jobs=-1)
        #print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
        if(mean(scores) <= best_mean):
            best_mean = mean(scores)
            best_weight = c
        results.append(scores)
    results.clear()
print("Penaltyenalty: elsticnet")
print("Best mean: ", best_mean)
print("Best weight: ", best_weight) 
print()
"""
#pyplot.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
#pyplot.show()

