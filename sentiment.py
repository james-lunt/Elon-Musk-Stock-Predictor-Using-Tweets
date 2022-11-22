import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

##### Reading in the Data #####
df = pd.read_csv(r"C:\Users\Niall Groves\Desktop\MachineLA\GroupAssignment\elonTestData.csv")
x = df.iloc [:,0]
y = df.iloc [:,1]

##### Getting Median of the Likes #####
median = np.median(y)
print("Median: ", median)

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
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X_train_tf, y, test_size=0.2)

##### Training a Model #####
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xtrain, ytrain)
preds = model.predict(xtest)
from sklearn.metrics import classification_report
#print(classification_report(ytest, preds))
from sklearn.metrics import precision_score
#print(precision_score(ytest, preds))

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(ytest, preds)
print('roc_auc_score for Logistic Regression: ', roc_auc_score(ytest, preds))

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - Logistic Regresion')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()