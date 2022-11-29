import numpy as np
import pandas as pd
import csv
import statistics as s
pd.options.mode.chained_assignment = None  # default='warn'

##### Train Sentiment Value model #####
df = pd.read_csv("elon_musk_tweets.csv", skiprows=0)
dates =df.iloc[:,0]
tweet_contents = df.iloc [:,1]
likes = df.iloc [:,2]
retweets = df.iloc[:,3]

#split = [10,100,1000]
#for n in split:
n = 10
quantiles_likes = s.quantiles(likes, n=n)
print("Quantiles: " + str(quantiles_likes))

n = len(quantiles_likes)-1
print(n)
#Weight retweet between 0 and 2 based on likes
for j in range(len(retweets)):
    for i in range(n):
        if i == n:
            retweets[j] = retweets[j]*2
            print(2)
            break
        elif (likes[j] > quantiles_likes[i] and likes[j] < quantiles_likes[i+1]):
            print((i/n)*2)
            retweets[j] = retweets[j]*((i+2/n)*2)
            #print(retweets[j])
            break

### TODO Need to fix when like is in lowest interquantile range as it's setting to 0

## TODO After adding retweets x weight and likes, sort array 

"""
i = 0
for num in likes:
    if likes[i] >= median:
        likes[i] = 1
    else:
        likes[i] = 0
    i = i + 1
i = 0
"""

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




