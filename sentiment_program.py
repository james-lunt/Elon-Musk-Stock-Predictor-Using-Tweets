from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import statistics as s
import time
pd.options.mode.chained_assignment = None  # default='warn'

today = "2022-11-15"

def read_tweets(name):
    df = pd.read_csv(name, skiprows=0)
    df = df.iloc[::-1]
    return df.iloc[:,0], df.iloc [:,1], df.iloc [:,2], df.iloc[:,3]

def convert_to_timestamp(dates):
    for i in range(len(dates)):
        dates[i] = time.mktime(time.strptime(dates[i], '%Y-%m-%d'))
        return dates


def weight_retweets(retweets,likes,quantiles):
    quantiles_likes = s.quantiles(likes, n=quantiles)

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
    return retweets


def assign_sentiments(likes,retweets):
    # Now add likes and weighted retweets to equal a sentiment value 
    sentiments = likes
    for i in range(len(likes)):
        sentiments[i] = likes[i] + retweets[i] 
    return sentiments


# Classify sentiment value based on interquantile
def label_sentiment(sentiments,quantiles):
    quantiles_sentiments = s.quantiles(sentiments, n=quantiles)
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
    return sentiments


def create_daily_sentiment(sentiments,dates):
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

    #Create dictionary of average sentiment values for each date
    sentiments_data_set = dict()
    for i in range(len(dates)):
        sentiments_data_set[dates[i]] = sentiments[i]

    return sentiments_data_set


def create_bag_of_words(tweet_contents):
    # Getting Rid of Stop Words
    # Creating a bag of Words
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer(stop_words='english')          #brought down to 527 from 638
    X_train_counts = count_vect.fit_transform(tweet_contents)

    # Tones down the weight applied to very frequent words
    from sklearn.feature_extraction.text import TfidfTransformer
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    
    return X_train_tf


def get_test_size(dates):
    count = 0
    for date in dates:
        if date == today:
            count+=1
    return count/len(dates)


def update_sentiments_with_todays_predictions(dates,sentiments,predictions):
    for i in range(len(dates)):
        if dates[i] == today:
            sentiments[i] = predictions[i]


def write_to_stock_predictor_training_set(stock_return_csv,sentiments):
    #Read CSV
    df = pd.read_csv(stock_return_csv, skiprows=0)
    stock_return_dates = df.iloc[:,0]
    stock_return = df.iloc [:,1]

    fields = ["Date", "Sentiment", "Stock Return"]
    with open("a_Dataset_Sentiment_Predictions.csv", 'w', newline='', encoding='utf-8') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile)     
        # writing the fields 
        csvwriter.writerow(fields)

        #Match date and sentiment value with stock return
        for i in range(len(stock_return_dates)):
            try:
                csvwriter.writerow([stock_return_dates[i], sentiments[stock_return_dates[i]], stock_return[i]])
            except:
                pass 


if __name__ == "__main__":
    
    # Returns the current local date
    #today = str(date.today())
    
    #Read dataset
    dates,tweet_content,likes,retweets = read_tweets("elon_musk_tweets.csv")

    #Convert Dates    
    #dates = convert_to_timestamp(dates)

    #Weigh retweets based on likes
    retweets = weight_retweets(retweets,likes,100)

    #Assign Sentimenet values combining likes and retweets
    sentiments = assign_sentiments(likes,retweets)

    #Label sentiment value by interquartile range
    sentiments = label_sentiment(sentiments,10)

    #Extract features from tweet text
    X = create_bag_of_words(tweet_content)
    xaxis = tweet_content.keys()
    print(type(X))

    #fig = plt.figure()
    plt.scatter(xaxis,sentiments)
    plt.show()
    #Stack input training data
    #X = np.column_stack((dates,X_words))

    #Train model with text extract as input and sentiment value as output
    #from sklearn.linear_model import LogisticRegression
    #sentiment_model = LogisticRegression(penalty = 'l2', C = 20, max_iter=1000)
    from sklearn.neighbors import KNeighborsRegressor
    sentiment_model = KNeighborsRegressor(n_neighbors=1,weights="uniform")
    #from sklearn.linear_model import LinearRegression
    #sentiment_model = LinearRegression()
    from sklearn.model_selection import train_test_split
    history_tweets, todays_tweets,history_sentiments, y_te = train_test_split(X,sentiments,shuffle=False,test_size=get_test_size(dates))
    #print(history_sentiments)    
    sentiment_model.fit(history_tweets,history_sentiments)

    #Predict sentiments for todays tweets
    todays_predictions = sentiment_model.predict(todays_tweets)
    print(todays_predictions)
    print(y_te)

    #Update sentiments with prediction sentiments
    sentiments = update_sentiments_with_todays_predictions(dates, sentiments, todays_predictions)

    #For sentiment scores that have the same dates then create daily average
    sentiments = create_daily_sentiment(sentiments,dates)

    #Using the stock return datasets, match dates and sentiment values to create stock predictor dataset
    write_to_stock_predictor_training_set("tesla_stock_return.csv",sentiments)

    #Predict sentiment and make daily sentiment
    #sentiment_predictions = create_daily_sentiment(sentiment_model.predict(X),dates)

    #Write todays data to training set
    #write_to_stock_predictor_training_set("tesla_stock_return.csv",sentiment_predictions)