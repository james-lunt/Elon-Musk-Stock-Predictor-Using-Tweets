import numpy as np
import pandas as pd
import csv
import math
pd.options.mode.chained_assignment = None  # default='warn'

##### Train Sentiment Value model #####
df = pd.read_csv("elon_musk_tweets.csv", skiprows=0)
dates =df.iloc[:,0]
tweet_contents = df.iloc [:,1]
likes = df.iloc [:,2]

# Getting Median of the Likes
median = np.median(likes)

# Assigning if Tweet is Positive or Negative Based on the Mean
i = 0
for num in likes:
    if likes[i] >= median:
        likes[i] = 1
    else:
        likes[i] = 0
    i = i + 1
i = 0

# If tweets land on the same day then set sentiment value to average sentiment value of tweets on that day
current_date = dates[0]
sentiments = []
daily_tweet_count = 0
for i in range(len(dates)):
    if dates[i] != current_date:
        for j in range(daily_tweet_count):
            likes[i-j-1] = round(np.average(sentiments))
        current_date = dates[i]
        sentiments =[]
        daily_tweet_count = 0
        daily_tweet_count += 1
        sentiments.append(likes[i])
    else:
        daily_tweet_count += 1
        sentiments.append(likes[i])


# Getting Rid of Stop Words
# Creating a bag of Words
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words='english')          #brought down to 527 from 638
X_train_counts = count_vect.fit_transform(tweet_contents)

# Tones down the weight applied to very frequent words
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)


# Training tweet sentiment model
from sklearn.linear_model import LogisticRegression

sentiment_model = LogisticRegression()
sentiment_model.fit(X_train_tf,likes)

## Train stock return model ##

#Parse stock dates and average daily stock return
df = pd.read_csv("tesla_stock_return.csv", skiprows=0)
stock_return_dates = df.iloc[:,0]
stock_return = df.iloc [:,1]

#Create dictionary of average sentiment values for each date
sentiments_data_set = dict()
for i in range(len(dates)):
    sentiments_data_set[dates[i]] = likes[i]


fields = ["Date", "Sentiment", "Stock Return"]
with open("Dataset.csv", 'w', newline='', encoding='utf-8') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile)     
    # writing the fields 
    csvwriter.writerow(fields)

    #Match date and sentiment value with stock return
    for i in range(len(stock_return_dates)):
        try:
            csvwriter.writerow([stock_return_dates[i], sentiments_data_set[stock_return_dates[i]], stock_return[i]])
        except:
            pass 




