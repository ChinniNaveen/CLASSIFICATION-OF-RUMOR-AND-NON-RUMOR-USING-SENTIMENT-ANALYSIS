# -*- coding: utf-8 -*-
import re 
import tweepy 
from tweepy import OAuthHandler 
from textblob import TextBlob 
import numpy as np
import json
import pandas as pd
from dateutil.parser import parse
import string

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

tokenizer = TweetTokenizer()
stemmer = PorterStemmer()
stopws = set(stopwords.words('english'))
vectorizer = CountVectorizer(analyzer = 'word')
class TwitterClient(object): 
    
    def clean_tweet(self, tweet): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
        #import PreProcessing
        #PreProcessing.preProcess(tweet)
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()) 

    def get_tweet_sentiment(self, tweet): 
        ''' 
        Utility function to classify sentiment of passed tweet 
        using textblob's sentiment method 
        '''
        # create TextBlob object of passed tweet text 
        analysis = TextBlob(self.clean_tweet(tweet)) 
        # set sentiment 
        if analysis.sentiment.polarity > 0: 
            return 'positive'
        elif analysis.sentiment.polarity == 0: 
            return 'neutral'
        else: 
            return 'negative'

    def get_tweets(self): 
        ''' 
        Main function to fetch tweets and parse them. 
        '''
        # empty list to store parsed tweets 
        tweets = []

        try: 
            # call twitter api to fetch tweets 
            #fetched_tweets = self.api.search(q = query, count = count) 
            tweets_file = open('twitter_data.txt', "r")
            print(tweets_file)
            # parsing tweets one by one

            for tweet in tweets_file: 
                
                # empty dictionary to store required params of a tweet 
                try:
                    data = {} 

                    tweett = json.loads(tweet)
                    
                    tweet =tweett.get("text")
                    
                    data['retweet_count']  = tweett.get('retweet_count')
                    data['favorite_count']  = tweett.get('favorite_count')
                    
                    dt = parse(tweett.get("user")['created_at'])

                    data['weekday'] = dt.weekday()
                    
                    data['month'] =  dt.month
                    
                    data['text']=tweet
                   
                    
                    
                    
                    data['response'] = False

                    if " @" in tweet or "@" in tweet: 
                        # if tweet has retweets, ensure that it is appended only once 
                        data['response'] = True
                    else: 
                        data['response'] = False
                    
                    tweets.append(data)


                except:
                    continue
            # return parsed tweets 
           

        except tweepy.TweepError as e: 
            # print error (if any) 
            print("Error : " + str(e)) 
        return tweets    
    
    
def clean_data(data):
    data.creationDate = pd.to_datetime(data.creationDate)
    data['creationDay'] = data.creationDate.apply(lambda x: x.weekday())
    data['creationMonth'] = data.creationDate.apply(lambda x: x.month)
    data.text = data.text.apply(tokenizer.tokenize)
    data.text = data.text.apply(lambda x: [w for w in x if w not in stopws and w.lower() not in string.punctuation])
    data.text = data.text.apply(lambda x: [stemmer.stem(w) for w in x])
    data['text_j'] = data.text.apply(lambda x: ' '.join(x))
    data['response'] = data.text_j.apply(lambda x: x.startswith(('@', ' @')))
    return data


def main(): 
    tweets = pd.read_csv('tweets_unfiltered.csv', delimiter=';')
    tweets = clean_data(tweets)
    
    attributes = tweets[['retweetCount', 'favoriteCount', 'creationDay', 'creationMonth', 'response']]
    print(type(attributes))
    #attributes = pd.concat((attributes, pd.DataFrame(vectorizer.fit_transform(tweets.text_j).toarray()[:200,:500])), axis=1)
    classes = tweets.rumor
    print(attributes)
    rf = RandomForestClassifier(n_estimators = 100)
    rf.fit(attributes, classes)

    # creating object of TwitterClient Class 
    api = TwitterClient() 
    # calling function to get tweets 
    tweets = api.get_tweets() 
    
    tweets = pd.DataFrame(tweets)
    
    print(tweets)

    attributes = tweets[['retweet_count', 'favorite_count', 'weekday', 'month', 'response']]

    
    pred  = rf.predict(attributes)


    print(pred)
if __name__ == "__main__": 
    # calling main function 
    main() 

