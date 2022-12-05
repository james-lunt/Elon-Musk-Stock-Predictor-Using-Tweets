# Elon-Musk-Stock-Predictor-Using-Tweets
Group Project for TCD Module CS7CS4: Machine Learning.

TSLA.csv is a dataset of daily stock return taken from Yahoo Fainance. <br />
The data points range from the 16th of November 2017 to the 16th of November 2022. Skipping Weekends <br />
The dataset is parsed into tesla_stock_return.csv with the script tesla_daily_average_stock_return.py <br />

The script get_elon_tweets.py scrapes Elon Musks tweets for the same dates as mentioned above, returning content, likes count and retweet count. <br />

sentiment_program.py then creates a dataset of tweets and sentiment values to train a  logistic regression model to predict sentiment values on tweet content.<br />
The script then performs prediction on a test slice of the tweets dataset which it was not used to train with. This is called dataset_sentiment_predictions.csv.<br />

There are three models that train and make predictions respectively in lasso.py, dummy_regressor.py and stock_program_Knn.py <br />

stock_program_Knn.py performs the best and we recommend using this one.


