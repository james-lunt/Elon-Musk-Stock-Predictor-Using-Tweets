import numpy as np
import pandas as pd
import csv
import statistics as s
import time
pd.options.mode.chained_assignment = None  # default='warn'

##### Train Sentiment Value model #####
df = pd.read_csv("elon_musk_tweets_test.csv", skiprows=0)
dates =df.iloc[:,0]
tweet_contents = df.iloc [:,1]
likes = df.iloc [:,2]
retweets = df.iloc[:,3]

#split = [10,100,1000]
#for n in split:
n=100
quantiles_likes = s.quantiles(likes, n=n)
print("Quantiles: " + str(quantiles_likes))


#Weight retweet between 0 and 2 based on likes
n = len(quantiles_likes)
weight_scores_size = n+1

index = 0
for like in likes:
    if like <= quantiles_likes[0]:
        retweets[index] = retweets[index]*((1/weight_scores_size)*2)
    elif like >= quantiles_likes[n-1]:
        retweets[index] = retweets[index]*(2)
    else:
        count = 0
        for quantile_like in quantiles_likes:
            if like <= quantiles_likes[count]:
                retweets[index] = retweets[index]*(((count+1)/weight_scores_size)*2)
                break
            count +=1
    index+=1

# Now add likes and weighted retweets to equal a sentiment value 
sentiments = likes
for i in range(len(likes)):
    sentiments[i] = likes[i] + retweets[i] 


# Find interquantile range
#split = [10,100,1000]
#for n in split:
n = 10
quantiles_sentiments = s.quantiles(sentiments, n=n)
print("Quantiles: " + str(quantiles_sentiments))

# Classify sentiment value based on interquantile
n = len(quantiles_sentiments)
weight_scores_size = n+1

index = 0
for sentiment in sentiments:
    if sentiment <= quantiles_sentiments[0]:
        sentiments[index] = 1
    elif sentiment >= quantiles_sentiments[n-1]:
        sentiments[index] = weight_scores_size
    else:
        count = 0
        for quantile_sentiment in quantiles_sentiments:
            if sentiment<= quantiles_sentiments[count]:
                sentiments[index] = count+1 
                break
            count +=1
    index+=1



# If tweets land on the same day then set sentiment value to average sentiment value of tweets on that day
current_date = dates[0]
sentiment_array = []
daily_tweet_count = 0
for i in range(len(dates)):
    if dates[i] != current_date:
        for j in range(daily_tweet_count):
            sentiments[i-j-1] = round(np.average(sentiment_array))
        current_date = dates[i]
        sentiment_array =[]
        daily_tweet_count = 0
        daily_tweet_count += 1
        sentiment_array.append(sentiments[i])
    else:
        daily_tweet_count += 1
        sentiment_array.append(sentiments[i])


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
sentiment_model.fit(X_train_tf,sentiments)

## Train stock return model ##

#Parse stock dates and average daily stock return
df = pd.read_csv("tesla_stock_return_Test.csv", skiprows=0)
stock_return_dates = df.iloc[:,0]
stock_return = df.iloc [:,1]

#Create dictionary of average sentiment values for each date
sentiments_data_set = dict()
for i in range(len(dates)):
    sentiments_data_set[dates[i]] = sentiments[i]


fields = ["Date", "Sentiment", "Stock Return"]
with open("Dataset_Test.csv", 'w', newline='', encoding='utf-8') as csvfile: 
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

