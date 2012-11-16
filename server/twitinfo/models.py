from django.db import models
from django import forms

class Tweet(models.Model):
    id = models.AutoField(primary_key=True, db_column="__id")
    tweet = models.TextField(db_column="text")
    user_id = models.IntegerField(default=0)
    screen_name = models.TextField(default="")
    created_at = models.DateTimeField('created date', db_index=True)
    sentiment = models.FloatField(db_column='sent')
    profile_image_url = models.TextField()
    location = models.TextField(null=True)
    latitude = models.FloatField(null=True)
    longitude = models.FloatField(null=True)

    class Meta:
        db_table = 'tweets_from_keywords'
    def __unicode__(self):
        return self.tweet

class Keyword(models.Model):
    key_word = models.CharField(max_length=200, unique=True)
    tweets = models.ManyToManyField(Tweet)
    max_indexed = models.IntegerField(default=-1)
    
    
    @staticmethod   
    def normalize_keywords(keywords):
        """
            takes in a comma separated string of keywords and returns a list of normalized keywords.
        """
        list_keywords=[]
        k=keywords.split(',')
        for key in k:
            filtered_key = Keyword.normalize(key)
            list_keywords.append(filtered_key)
        return list_keywords

    @staticmethod
    def normalize(kw):
        return (' '.join(kw.split())).lower()
        
    def __unicode__(self):
        return self.key_word

class WordFrequency(models.Model):
    word = models.CharField(max_length=300, unique=True)
    idf = models.FloatField()
    
class Event(models.Model):
    featured = models.BooleanField(default=False)
    name = models.CharField(max_length=200)
    start_date = models.DateTimeField(null=True, blank=True)
    end_date = models.DateTimeField(null=True, blank=True)
    keywords = models.ManyToManyField(Keyword)
    children = models.ManyToManyField("self", symmetrical=False, related_name='parents', blank=True)

    def __unicode__(self):
        return self.name
    
    @staticmethod   
    def normalize_name(name):
        return (' '.join(name.split())).lower()

class TweetFrequency(models.Model):
    event = models.ForeignKey(Event)
    tweet_date = models.DateTimeField(db_index=True)
    interval = models.IntegerField(db_index=True)
    num_tweets = models.IntegerField(default=0)
    pos_sentiment = models.FloatField()
    neg_sentiment = models.FloatField()

    class Meta:
        unique_together = ("event", "tweet_date", "interval")

class Peak(models.Model):
    event = models.ForeignKey(Event)
    peak_date = models.DateTimeField(db_index=True)
    start_date = models.DateTimeField(db_index=True)
    end_date = models.DateTimeField()
    start_freq = models.IntegerField(default=0)
    interval = models.IntegerField()
    freqwords = models.CharField(max_length=1500)

    class Meta:
        unique_together = ("event", "start_date", "interval")
    
class EventStats(models.Model):
    event = models.ForeignKey(Event)
    pos = models.IntegerField() #-1 for last, -2 for second last tweet group
    num_points = models.IntegerField(default=0)
    ewma = models.FloatField(default=0.0) # exponentially weighted moving avgerage
    ewmmd = models.FloatField(default=0.0) # exponentially weighted moving mean deviation
    num_tweets = models.IntegerField(default=0)
    tweet_date = models.DateTimeField()
    interval = models.IntegerField()
    is_peak = models.BooleanField(default=False)

    class Meta:
        unique_together = ("event", "pos", "interval")

    def copy_data_from(self, src):
        self.num_points = src.num_points
        self.ewma = src.ewma
        self.ewmmd = src.ewmmd
        self.num_tweets = src.num_tweets
        self.tweet_date = src.tweet_date
        self.interval = src.interval
        self.is_peak = src.is_peak
