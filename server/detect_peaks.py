import settings
from django.core.management import setup_environ
setup_environ(settings)
from twitinfo.models import Event, Tweet, Keyword, WordFrequency, Peak, EventStats, TweetFrequency
from threading import Timer

from django.db import transaction
from datetime import datetime,timedelta
from django.db.models import Avg, Max, Min, Count, Sum, F
from tweeql.builtin_functions import MeanOutliers
import nltk
from django.core.cache import cache
from django.db import connection, reset_queries

import threading
import traceback
import time
import sys
import logging
import json

#formatter = logging.Formatter('%(asctime)s | %(name)15s:%(lineno)5s | %(levelname)10s | %(message)s')
formatter = logging.Formatter('%(asctime)s | %(lineno)5s | %(message)s')
CH = logging.StreamHandler()
CH.setFormatter(formatter)
logger = logging.getLogger('detect_peaks.py')
#logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)
logger.addHandler(CH)
detail_logger = logging.getLogger('detect_peaks.py-details')
detail_logger.setLevel(logging.INFO)
detail_logger.addHandler(CH)

CACHE_SECONDS = 864000 # 10 days
REFRESH_SECONDS = 5    # default wait time between detect_loop's

# Dicts needed for datetime calculation and convertion.
tdelta =          { 120: timedelta(minutes=2),
                    600: timedelta(minutes=10),
                    3600: timedelta(hours=1),
                    86400: timedelta(days=1)}

window =          { 120: timedelta(days=1),
                    600: timedelta(days=5),
                    3600: timedelta(days=30),
                    86400: timedelta(days=3652)} # roughly 10 years

source_interval = { 600: 120,
                    3600: 600,
                    86400: 3600}

stf =             { 120: {"date": ('%Y,%m,%d,%H,%M'),"d": 'new Date(%Y,%m-1,%d,%H,%M)'},
                    600: {"date": ('%Y,%m,%d,%H,%M'),"d": 'new Date(%Y,%m-1,%d,%H,%M)'},
                    3600: {"date": ('%Y,%m,%d,%H'),   "d": 'new Date(%Y,%m-1,%d,%H)'},
                    86400: {"date": ('%Y,%m,%d'),      "d": 'new Date(%Y,%m-1,%d)'}}

##if settings.DATABASES['default']['ENGINE'] == 'django.db.backends.postgresql_psycopg2':
##    select_data = { 120: {"d":    "to_char(created_at, 'ne\"w\" \"D\"ate(YYYY,MM-1,DD, HH24,MI)')",
##                          "date": "to_char(created_at, 'YYYY,MM,DD,HH24,MI')"},
##                    600: {"d":    "to_char(tweet_date, 'ne\"w\" \"D\"ate(YYYY,MM-1,DD, HH24,MI)')",
##                          "date": "to_char(tweet_date, 'YYYY,MM,DD,HH24,MI')"},
##                    3600: {"d":    "to_char(tweet_date, 'ne\"w\" \"D\"ate(YYYY,MM-1,DD,HH24)')",
##                           "date": "to_char(tweet_date, 'YYYY,MM,DD,HH24')"},
##                    86400: {"d":    "to_char(tweet_date, 'ne\"w\" \"D\"ate(YYYY,MM-1,DD)')",
##                            "date": "to_char(tweet_date, 'YYYY,MM,DD')"}}
##else:
##    select_data = { 120: {"d":    "strftime('new Date(%%Y,%%m-1,%%d,%%H,%%M)', created_at)",
##                          "date": "strftime(('%%Y,%%m,%%d,%%H,%%M') , created_at)"},
##                    600: {"d":    "strftime('new Date(%%Y,%%m-1,%%d,%%H,%%M)', tweet_date)",
##                          "date": "strftime(('%%Y,%%m,%%d,%%H,%%M') , tweet_date)"},
##                    3600: {"d":    "strftime('new Date(%%Y,%%m-1,%%d,%%H)', tweet_date)",
##                           "date": "strftime(('%%Y,%%m,%%d,%%H') , tweet_date)"},
##                    86400: {"d":    "strftime('new Date(%%Y,%%m-1,%%d)', tweet_date)",
##                            "date": "strftime(('%%Y,%%m,%%d') , tweet_date)"}}        

#postgres: to epoch: floor(extract('epoch' from created_at)/<interval>)*<interval>
#          reverse:  timestamp 'epoch' + <epoch> * interval '1 second'
#sqlite:   to epoch: strftime('%s', created_at)/<interval>*<interval> # does not work for django because .extra() tries to format %%s
#          to epoch: cast((julianday(created_at) - 2440587.5)*86400/<interval> as int)*<interval>
#          reverse:  datetime(<epoch>, 'unixepoch')
#mysql:    to epoch: floor(unix_timestamp(created_at)/<interval>)*<interval>
#          reverse:  from_unixtime(<epoch>)

select_data = dict()
for interval in tdelta.keys():
    column = 'created_at' if interval == 120 else 'tweet_date'
    if settings.DATABASES['default']['ENGINE'] == 'django.db.backends.postgresql_psycopg2':
        select_data[interval] = {"date" : "floor(extract('epoch' from %s)/%s)*%s" % (column, interval, interval)}
    elif settings.DATABASES['default']['ENGINE'] == 'django.db.backends.sqlite3':
        select_data[interval] = {"date" : "cast((julianday(%s) - 2440587.5)*86400/%s as int)*%s" % (column, interval, interval)}

def custom_query(query, params):
    cursor = connection.cursor()
    cursor.execute(query, params)
    return [dict(zip([col[0] for col in cursor.description], row))
            for row in cursor.fetchall()]

def query_tweets(interval, event_id, sdate, edate):
    sql = 'SELECT floor(extract(\'epoch\' from created_at)/%s)*%s AS "date", '       \
                 'sum(case when sent > 0 then sent else 0 end) AS "pos_sentiment", ' \
                 'sum(case when sent < 0 then sent else 0 end) AS "neg_sentiment", ' \
                 'count(__id) AS "num_tweets" '                                      \
          'FROM "tweets_from_keywords" '                                             \
          'WHERE __id IN '                                                           \
               '(SELECT tweet_id FROM "twitinfo_keyword_tweets" '                    \
                'WHERE keyword_id IN '                                               \
                    '(SELECT keyword_id FROM "twitinfo_event_keywords" '             \
                     'WHERE event_id = %s)) '                                        \
          'AND created_at between %s and %s '                                        \
          'GROUP BY date '                                                           \
          'ORDER BY date ASC '
    params = (interval, interval, event_id, sdate, edate)
    print sql % params
    return custom_query(sql, params)

def query_locations(event_id, sdate, edate):
    params = (interval, interval, event_id, sdate, edate)
    print locations_sql % params
    return custom_query(locations_sql, params)

def dt2ts(dt):
    return int(time.mktime(dt.timetuple()))

def ts2dt(ts):
    return datetime.utcfromtimestamp(ts)

def json_handler(obj):
    if isinstance(obj, datetime):
        return 'new Date(%s)' % str(dt2ts(obj))
    raise TypeError

def convert_date(date):
    d=date.split(',')
    d=map(int , d)
    dt=datetime(*d)
    return dt

def convert_date(ts):
    return ts2dt(ts)

def floor_date(date, tdelta):
    total_sec = tdelta.total_seconds()
    if total_sec >= 60:
        date = date.replace(microsecond=0, second=0)
    if total_sec >= 60 * 60:
        date = date.replace(minute=0)
    if total_sec >= 24 * 60 * 60:
        date = date.replace(hour=0)
    return date

def find_end_dates(tweets,list_peaks):
    i = 0 # index for list_peaks
    k = 0 # index for tweets
    len_peaks = len(list_peaks)
    len_tweets = len(tweets)

    if not len_peaks:
        return list_peaks;

    for i in range(len_peaks):
        # find a tweet with date matching current peak 
        for j in range(len_tweets):
            if(list_peaks[i]["start_date"] == tweets[j]['date']):
                k = j+1
                break

        # starting from the subsequent tweet
        while(k < len_tweets):
            # if current tweet's date equals next peak's start date               OR
            #    current tweet's has <= #tweets than current peak's start freq    OR
            #    this is the last tweet
            if((i+1 < len_peaks and tweets[k]['date'] == list_peaks[i+1]['start_date'])  or
               tweets[k]['num_tweets'] <= list_peaks[i]["start_freq"] or
               k == len_tweets-1):
                list_peaks[i]["end_date"] = tweets[k]['date']
                if list_peaks[i]["start_date"] == list_peaks[i]["end_date"]:
                    logger.error('Peak %s has same end date as start date. Source tweet freq: %s', list_peaks[i], tweets[k])
                break
            k+=1
    
    return list_peaks  

def words_by_tfidf(dic, keywords):
    detail_logger.debug('Entering words_by_tfidf')
    freq_words = []

    temp_words = WordFrequency.objects.all().values_list('word', 'idf')
    for word, idf in temp_words:
        if dic.has_key(word):
            freq_words.append([word, idf])

    detail_logger.debug('words_by_tfidf returns %s frequent words among %s words', len(freq_words), len(dic))

    # multiply by -1*idf if it exists in the idf list.  record the largest
    # idf witnessed
    maxidf = 0
    for word, idf in freq_words:
        if word in dic:
            dic[word] *= -1 * idf
            if idf > maxidf:
                maxidf = idf
    
    # for all idfs which existed, make the tfidf positive again.
    # for all already-positive idfs (which didn't have an idf for this
    # word), multiply by 10 more than the largest idf in the list.
    maxidf += 10
    for word, idf in dic.items():
        if idf < 0:
            dic[word] *= -1
        else:
            dic[word] *= maxidf
    
    words = dic.keys() 
    words.sort(cmp=lambda a,b: cmp(dic[b],dic[a]))

    detail_logger.debug('Exiting words_by_tfidf')
    return words

def find_max_terms(tweets, keywords, stopwords):
    total_freq=0
    words_dict={}

    for tweet in tweets:    
        text = tweet['tweet']
        tweet_text=text.lower().split()      
        for word in tweet_text:
            if word in stopwords:
                continue
            if words_dict.has_key(word):
                words_dict[word]+=1
            else:
                words_dict[word]=1

    detail_logger.debug('Populated tweet dictionary of %s words', len(words_dict))
    return words_by_tfidf(words_dict, keywords)
 
def annotate_peaks(peaks,tweets,event_id,keywords):
    detail_logger.debug('Entering annotate_peaks')
    list_keywords = ", ".join(keywords)
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stopwords.add("rt")
    stopwords.add("...")
    stopwords.add("&amp;")
    stopwords.add("-")
    for keyword in keywords:
        stopwords.add(keyword)
    detail_logger.debug('Retrieved %s stopwords', len(stopwords))

    for peak in peaks:
        sdt = peak['start_date']
        edt = peak['end_date']
        t = Tweet.objects.filter(keyword__event = event_id, created_at__gte = sdt, created_at__lte=edt).values('tweet').iterator()
        sorted_list = find_max_terms(t, keywords, stopwords)
        peak_tweet = peak['peak_tweet']
        peak_tweet['title']=", ".join(sorted_list[:5])
        peak_tweet['data']={'event':event_id,'keywords':list_keywords,'start_date':sdt,'end_date':edt}
    return tweets

def save_stats(tweets, i, event, interval, stats, mean_outliers, *group):
    l = len(tweets)
    if i == l-1:
        if stats.has_key(-1):
            if not stats.has_key(-2):
                stats[-2] = EventStats(event = event, pos = -2, interval = interval)
            stats[-2].copy_data_from(stats[-1])
        else:
            stats[-1] = EventStats(event = event, pos = -1, interval = interval)
        stats[-1].num_points, stats[-1].ewma, stats[-1].ewmmd, = mean_outliers.export_stats(*group)
        stats[-1].num_tweets = tweets[i]['num_tweets']
        stats[-1].tweet_date = tweets[i]['date']
        stats[-1].is_peak = False
        logger.debug('Found last tweet group of i=%s: %s', i, tweets[i])
        logger.debug([stats[i].__dict__ for i in stats])
    elif i == l-2:
        if not stats.has_key(-1):
            stats[-1] = EventStats(event = event, pos = -1, interval = interval)
        # populate second last stats to stats[-1], which will be rolled over to stats[-2]
        # when processing the last stats.
        stats[-1].num_points, stats[-1].ewma, stats[-1].ewmmd, = mean_outliers.export_stats(*group)
        stats[-1].num_tweets = tweets[i]['num_tweets']
        stats[-1].tweet_date = tweets[i]['date']
        stats[-1].is_peak = False
        logger.debug('Found second last tweet group of i=%s: %s', i, tweets[i])
        logger.debug([stats[i].__dict__ for i in stats])

def peak_child_detection(children,list_peaks,tweets):
    list_overlaps=[]
    i=0
    savej=0
    count=0
    while(i<len(children)):
        j=savej
        while(j<len(list_peaks)):
            
            if ((children[i].start_date >= list_peaks[j]['start_date'] and children[i].start_date <= list_peaks[j]['end_date']) or
                (children[i].end_date >= list_peaks[j]['start_date']  and children[i].end_date <= list_peaks[j]['end_date']) or
                    (list_peaks[j]['start_date'] >= children[i].start_date and list_peaks[j]['start_date'] <= children[i].end_date) or
                    (list_peaks[j]['end_date'] >= children[i].start_date and list_peaks[j]['end_date'] <= children[i].end_date)):
                   
                    if(count==1):
                        savej=j-1
                    
                    if list_peaks[j].has_key("children"):   
                        list_peaks[j]["children"].append(children[i])
                    else:
                        list_peaks[j]["children"]=[children[i]] 
                    
                    j+=1
                
            elif(list_peaks[j]['start_date'] > children[i].end_date):
                j=savej
                break
            elif(list_peaks[j]['end_date'] < children[i].start_date):
                count+=1
                j+=1
            
        i+=1
       
    j=0
    for peak in list_peaks:
        if peak.has_key("children"):
            while(j<len(tweets)):
                if tweets[j]['date'] == peak["peak_date"]:
                    tweets[j]['children']=peak["children"]
                    j+=1
                    break
                else:
                    tweets[j]['children'] = None
         
                j+=1
      
  
    while(j<len(tweets)):
        tweets[j]['children'] = None
        j+=1
    
    return tweets

@transaction.commit_manually
def detect_peaks(event_id, interval):
    def commit_changes(return_val = None):
        try:
            transaction.commit()
        except:
            raise
        return return_val
        
    logger.info('Entering detect_peaks for event id %s, interval %s', event_id, interval)

    e = Event.objects.get(id=event_id)
    sdate = e.start_date
    edate = e.end_date
    tweets = Tweet.objects.filter(keyword__event = event_id)
    keywords = Keyword.objects.filter(event__id = event_id)
    last_tweet_id = max([kw.max_indexed for kw in keywords])

    if last_tweet_id < 0:
        logger.debug('No tweets for this event yet')
        return commit_changes()

    last_tweet = Tweet.objects.filter(id = last_tweet_id).values()[0]
    key_last_tweet = 'event' + str(event_id) + 'interval' + str(interval) + 'last_tweet' 
    prev_last_tweet = cache.get(key_last_tweet)

    if prev_last_tweet:
        if prev_last_tweet['id'] == last_tweet_id:
            logger.debug('No new tweets for this event')
            return commit_changes()

        # to avoid excessive calculation, do not detect peaks unless there is enough new tweets
        if (last_tweet['created_at'] - prev_last_tweet['created_at']).total_seconds() < REFRESH_SECONDS:
            logger.debug('Not enough new tweets for this event')
            return commit_changes()

    #sanity check
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Start sanity check')
        try:
            tf = TweetFrequency.objects.filter(event = e, interval = interval).aggregate(max=Max('tweet_date'))['max']
            tf = TweetFrequency.objects.get(event = e, interval = interval, tweet_date = tf) if tf else None
        except:
            tf = None
        try:
            es = EventStats.objects.get(event = e, pos = -1, interval = interval)
        except:
            es = None
        
        if es and tf:
            if es.tweet_date != tf.tweet_date or es.num_tweets != tf.num_tweets:
                logger.error('Mismatched TF: [%s %s], ES [%s %s]',
                             tf.tweet_date, tf.num_tweets,
                             es.tweet_date, es.num_tweets)
                raise

            if es.is_peak:
                try:
                    Peak.objects.get(event = e, peak_date = es.tweet_date, interval = interval)
                except:
                    logger.error('Missing peak -1: [%s %s]', es.tweet_date, interval)
                    raise

        try:
            es = EventStats.objects.get(event = e, pos = -2, interval = interval)
        except:
            es = None

        if es and es.is_peak:
            try:
                Peak.objects.get(event = e, peak_date = es.tweet_date, interval = interval)
            except:
                logger.error('Missing peak -2: [%s %s]', es.tweet_date, interval)
                raise
            
        logger.debug('End sanity check')

    hist_stats = dict()
    try:
        hist_stats[-1] = EventStats.objects.get(event__id = e.id, interval = interval, pos = -1)
        hist_stats[-2] = EventStats.objects.get(event__id = e.id, interval = interval, pos = -2)
    except:
        pass
    logger.debug('Hist stats, -2: %s, -1: %s',
                 hist_stats[-2].__dict__ if hist_stats.has_key(-2) else None,
                 hist_stats[-1].__dict__ if hist_stats.has_key(-1) else None)

    if hist_stats.has_key(-1):
        sdate = hist_stats[-1].tweet_date
    if sdate == None:
        key_first_date = 'event' + str(event_id) + 'first_date'
        sdate = cache.get(key_first_date)
        if sdate == None:
            sdate = tweets.aggregate(min=Min('created_at'))['min']
            logger.debug('sdate is %s', sdate)
            if sdate == None:
                logger.debug(connection.queries)
            cache.set(key_first_date, sdate, CACHE_SECONDS)
    if edate == None:
        edate = last_tweet['created_at']
    logger.debug('Detecting peaks in tweets date range: [%s, %s]', sdate, edate)

    if interval == 120:
        #tweets = tweets.filter(created_at__gte = sdate, created_at__lte = edate).extra(select = select_data[interval]).values('date').annotate(num_tweets = Count('tweet')).order_by('date')
        try:
            tweets = query_tweets(interval, event_id, sdate, edate)
        except:
            traceback.print_exc()
            raise
    else:
        try:
            tweets = TweetFrequency.objects.filter(event__id = event_id, interval = source_interval[interval])
            tweets = tweets.filter(tweet_date__gte = sdate, tweet_date__lte = edate).extra(select = select_data[interval]).values('date')
            tweets = tweets.annotate(num_tweets = Sum('num_tweets'), pos_sentiment = Sum('pos_sentiment'), neg_sentiment = Sum('neg_sentiment')).order_by('date')
        except:
            traceback.print_exc()
            raise
    logger.debug('a')

    logger.debug(len(tweets))
    tweets = list(tweets)
    for tweet in tweets:
        tweet['date'] = convert_date(tweet['date'])
    logger.debug('DB retrieves %s tweet groups', len(tweets))

#    logger.debug('DB queries: %s', connection.queries)
    reset_queries()
    if interval > 600:
        logger.debug(tweets)

    if len(tweets) == 0:
        return commit_changes()

    mean_outliers = MeanOutliers()
    detector = mean_outliers.nummeandevs

    saved_peak_date = None
    if hist_stats.has_key(-1) and hist_stats[-1].is_peak:
        saved_peak_date = hist_stats[-1].tweet_date
    elif hist_stats.has_key(-2) and hist_stats[-2].is_peak:
        saved_peak_date = hist_stats[-2].tweet_date

    last_stats = None
    if hist_stats.has_key(-1):
        # If the first tweet group has the same num_tweets as previously seen,
        # do not process it again.
        if hist_stats[-1].num_tweets == tweets[0]['num_tweets']:
            tweets.pop(0)
            last_stats = hist_stats[-1]
            if len(tweets) == 0:
                cache.set(key_last_tweet, last_tweet, CACHE_SECONDS)
                return commit_changes()
        elif hist_stats.has_key(-2):
            last_stats = hist_stats[-2]
            #ctong: stats[-1] no longer valid, populate with previous stats stats[-2]
            hist_stats[-1].copy_data_from(hist_stats[-2])
        if last_stats:
            logger.debug(last_stats.__dict__)
            mean_outliers.import_stats([last_stats.num_points, last_stats.ewma, last_stats.ewmmd], 1)

    logger.debug('MeanOutliers stats: %s', mean_outliers.export_stats(1))

    ## ctong: the last tweetfreq from previous run might be the same as the first tweetfreq from current run.
    first_tweetfreq_idx = 0
    last_tweetfreq = None
    try:
        last_tweetfreq = TweetFrequency.objects.get(event = e, interval = interval, tweet_date = tweets[0]['date'])
    except:
        pass
    if last_tweetfreq:
        last_tweetfreq.num_tweets = tweets[0]['num_tweets']
        last_tweetfreq.pos_sentiment = tweets[0]['pos_sentiment']
        last_tweetfreq.neg_sentiment = tweets[0]['neg_sentiment']
        last_tweetfreq.save()
        first_tweetfreq_idx = 1

    for i in range(first_tweetfreq_idx, len(tweets)):
        tweetfreq = tweets[i]
        t = TweetFrequency(event = e,
                           tweet_date = tweetfreq['date'],
                           interval = interval,
                           num_tweets = tweetfreq['num_tweets'],
                           pos_sentiment = tweetfreq['pos_sentiment'],
                           neg_sentiment = tweetfreq['neg_sentiment'])
        try:
            t.save()
        except:
            traceback.print_exc()
            print i
            print tweetfreq
            raise

    prev_tweet = {'title':'null','data':'null','children':'null'}
    if last_stats:
        prev_tweet['num_tweets'] = last_stats.num_tweets
        prev_tweet['date'] = last_stats.tweet_date
    else:
        prev_tweet['num_tweets'] = tweets[0]['num_tweets']
        prev_tweet['date'] = tweets[0]['date']
    tweets.insert(0, prev_tweet)

    i = 1
    list_peaks = []
    # loop through the tweets and detect a peak based on mean deviation function provided. save the start date
    # and the date of the peak in a dictionary.  save each peak in list_peaks.
    while i < len(tweets):
        tweets[i]['title'] = 'null'
        tweets[i]['data'] = 'null'
        sdt_p = tweets[i-1]['date']
        sdt_n = tweets[i]['date']
        delta_d = (sdt_n-sdt_p).total_seconds()/interval

        if delta_d > 1:
            j = 0
            while(j<delta_d-1):
                insert_tweet = {'title':'null','num_tweets':0,'data':'null','children':'null'}
                sdt_p += tdelta[interval]
                insert_tweet['date'] = sdt_p
                tweets.insert(i+j, insert_tweet)
                j += 1
            i += j
        i += 1

    i = 1

    while i < len(tweets):
        current_val = tweets[i]['num_tweets']
        previous_val = tweets[i-1]['num_tweets']
        mdiv = detector(None,tweets[i]['num_tweets'], 1)
        save_stats(tweets, i, e, interval, hist_stats, mean_outliers, 1)
        new_peak = (mdiv > 2.0 and current_val > 10)
        cont_peak = (i == 1 and saved_peak_date != None)

        if i==1:
            logger.debug('first tweet is new peak %s cont_peak %s', new_peak, cont_peak)
        
        if (cont_peak or new_peak) and current_val > previous_val:
            if cont_peak:
                logger.debug('d')
                try:
                    saved_peak = Peak.objects.get(event = e, peak_date = saved_peak_date, interval = interval)
                except:
                    traceback.print_exc()
                    raise
                start_freq = saved_peak.start_freq
                start_date = saved_peak.start_date
                logger.debug('e')
            elif new_peak:
                start_freq = previous_val 
                start_date = tweets[i-1]['date']                
            # once a peak is detected, keep climbing up the peak until the maximum is reached. store the peak date and keep
            # running the mdiv function on each value because it is requires previous values to calculate the mean.
            while(current_val > previous_val):
                  if i+1<len(tweets):
                      i += 1
                      mdiv = detector(None,tweets[i]['num_tweets'], 1)
                      save_stats(tweets, i, e, interval, hist_stats, mean_outliers, 1)
                      current_val = tweets[i]['num_tweets']
                      previous_val = tweets[i-1]['num_tweets'] 
                      peak_date = tweets[i-1]['date']
                      peak_tweet = tweets[i-1]
                  else:
                      peak_date = tweets[i]['date']
                      peak_tweet = tweets[i]
                      break
            #d = {"start_date":start_date,"start_freq":start_freq ,"peak_date":peak_date}
            d = {"start_date":start_date,"start_freq":start_freq ,"peak_date":peak_date, "peak_tweet":peak_tweet}
            list_peaks.append(d)
        i += 1
    tweets = tweets[1:]

    if len(list_peaks):
        if tweets[-1]['date'] == list_peaks[-1]['peak_date']:
            if hist_stats.has_key(-1):
                hist_stats[-1].is_peak = True
                hist_stats[-2].is_peak = False
        elif tweets[-2]['date'] == list_peaks[-1]['peak_date']:
            if hist_stats.has_key(-2):
                hist_stats[-2].is_peak = True

    logger.debug([hist_stats[i].__dict__ for i in hist_stats])
    logger.debug('MeanOutliers returns %s peak tweet groups', len(list_peaks))
    logger.debug('%s tweets', len(tweets))

    logger.debug('MeanOutliers stats: %s', mean_outliers.export_stats(1))
    
    keywords = Keyword.objects.filter(event__id = event_id)
    words = [kw.key_word for kw in keywords]
    try:
        peaks = find_end_dates(tweets,list_peaks)
    except:
        traceback.print_exc()

    logger.debug('find_end_dates returns %s peak tweet groups', len(peaks))

    try:
        tweets = annotate_peaks(peaks,tweets,event_id,words)
    except:
        traceback.print_exc()

    logger.debug('annotate_peaks returns %s tweets', len(tweets))
    logger.debug('annotate_peaks returns %s peak tweet groups', len(peaks))

    try:
        children = e.children.order_by('start_date')
        tweets = peak_child_detection(children,list_peaks,tweets)
    except:
        tweets = tweets

    logger.debug('peak_child_detection returns %s tweets', len(tweets))
    logger.debug('peak_child_detection returns %s peak tweet groups', len(list_peaks))

    if hist_stats.has_key(-1):           
        hist_stats[-1].save()

    if hist_stats.has_key(-2):
        hist_stats[-2].save()

    ## ctong: the last peak from previous run might be the same as the first peak from current run.
    first_peak_idx = 0
    last_peak = None
    if len(list_peaks):
        try:
            last_peak = Peak.objects.get(event = e, start_date = list_peaks[0]['start_date'], interval = interval)
        except:
            pass
    if last_peak:
        peak = list_peaks[0]
        last_peak.peak_date = peak['peak_date']
        last_peak.end_date = peak['end_date']
        last_peak.start_freq = peak['start_freq']
        last_peak.freqwords = peak['peak_tweet']['title']
        last_peak.save()
        first_peak_idx = 1

    logger.debug('first_peak_idx: %s', first_peak_idx)
    if first_peak_idx == 1:
        logger.debug('first peak: %s', last_peak.__dict__)
    
    # ctong bug: end date could be missing ...
    for i in range(first_peak_idx, len(list_peaks)):
        peak = list_peaks[i]
        p = Peak(event = e,
                 peak_date = peak['peak_date'],
                 start_date = peak['start_date'],
                 end_date = peak['end_date'],
                 start_freq = peak['start_freq'],
                 interval = interval,
                 freqwords = peak['peak_tweet']['title'])
        p.save()

    if len(list_peaks):
        logger.debug('last peak: %s', list_peaks[-1])

    tweets = []
    list_peaks = []

    cache.set(key_last_tweet, last_tweet, CACHE_SECONDS)
    logger.info('Exiting detect_peaks')

    return commit_changes(1)

def detect_loop():
    counter = 0
    last_detected = {}
    while True:
        events = Event.objects.all()
        awake = 0
        val = 0
        for event in events:
            now = datetime.now()
            if last_detected.has_key(event.id):
                if now - last_detected[event.id] < timedelta(seconds = REFRESH_SECONDS):
                    break
            last_detected[event.id] = now
            for interval in sorted(tdelta.keys()):
                try:
                    val = detect_peaks(event.id, interval)
                    reset_queries()
                    connection.close()
                except:
                    print 'Error 1', connection.queries
                    reset_queries()
                    traceback.print_exc()
                    raise
                if val:
                    awake += 1
        if not awake:
            time.sleep(REFRESH_SECONDS)
            counter += 1
        if counter == 1000:
            print "Ending detector to avoid memory leaks"
            break

# Prints thread stacks if you push ctrl+\
def dumpstacks(signal, frame):
    id2name = dict([(th.ident, th.name) for th in threading.enumerate()])
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append("\n# Thread: %s(%d)" % (id2name[threadId], threadId))
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
            if line:
                code.append("  %s" % (line.strip()))
    print "\n".join(code)

import signal
signal.signal(signal.SIGQUIT, dumpstacks)

if __name__ == "__main__":
    detect_loop()
