    # -*- coding: utf-8 -*-
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
access_token = "2158465844-TYtigqXGBQa5KzshKjz5MFO9SqTVkc03FQLL37N"
access_token_secret = "ClgQD5kCml8yb70ZlseUQIXOAM6bltOwoPuD5Z3SWguXW"
consumer_key = "IaIdBOJa7ZwNI6xHS11Jg5DVb"
consumer_secret = "IuZ4G5wrK2aYU1yuOwbPqu6G0Rx1hBgHwo6xEOHANhA4gTzJv9"
file2write=open("twitter_data.txt",'w')
i=1;
#no of tweets need to load 
no_tweets=200
lines=[]
#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):
   
    def on_data(self, data):
        global i
        global no_tweets
        
        if i>no_tweets:  
            return False
        else:      
            file2write.write(data)
            print("tweet %s Loaded..."%i)
            i+=1
            return True
        

    def on_error(self, status):
        print(status)


if __name__ == '__main__':
    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)
    #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
    stream.filter(languages=["en"],track=['Trump'])
    file2write.close()