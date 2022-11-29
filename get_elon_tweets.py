import snscrape.modules.twitter as sntwitter
import csv

#Query elon musks tweets from 16/11/2017 until 16/11/2022
query = "(from:elonmusk) until:2022-11-16 since:2017-11-16 -filter:replies"
csv_output = "elon_musk_tweets.csv"
fields = ['Date','Tweet','Likes','Retweets']

with open(csv_output, 'w', newline='', encoding='utf-8') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile)     
    # writing the fields 
    csvwriter.writerow(fields)

    #loop through each tweet
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        #Write tweet data, content, like count and retweet count
        csvwriter.writerow([tweet.date.date(),tweet.content,tweet.likeCount,tweet.retweetCount])
