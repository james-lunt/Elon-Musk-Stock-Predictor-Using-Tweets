import snscrape.modules.twitter as sntwitter
import csv

query = "(from:elonmusk) until:2022-11-16 since:2017-11-16 -filter:replies"
tweets = []
csv_output = "elon_musk_tweets.csv"
fields = ['Date','Tweet','Likes','Retweets']

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    tweets.append([tweet.date,tweet.content,tweet.likeCount,tweet.retweetCount])

print(tweets)
with open(csv_output, 'w', newline='') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
        
    # writing the data rows 
    csvwriter.writerows(tweets)
