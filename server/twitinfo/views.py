from django import forms
from django.shortcuts import get_object_or_404, render_to_response
from django.http import HttpResponseRedirect, HttpResponse, Http404
from django.core.cache import cache
from django.core.context_processors import csrf
from django.core.urlresolvers import reverse
from django.db.models import Avg, Max, Min, Count, Sum, F
from django.template import RequestContext, Context, Template, loader
from django.views.decorators.cache import cache_page
from server.twitinfo.models import Event,Tweet, Keyword, WordFrequency, Peak, EventStats, TweetFrequency
from datetime import datetime,timedelta
from operator import itemgetter
import itertools
import json
import nltk
import re
import random
import sys
import settings
from tweeql.builtin_functions import MeanOutliers
#ctong import
import time, traceback, logging, calendar
from django.db import connection # connection.queries return raw sql in debug mode
from django.db import reset_queries

#ctong Jinja2
import jinja2
from jinja2 import Environment, FileSystemLoader

#formatter = logging.Formatter('%(asctime)s | %(name)15s:%(lineno)5s | %(levelname)10s | %(message)s')
formatter = logging.Formatter('%(asctime)s | %(lineno)5s | %(message)s')
CH = logging.StreamHandler()
CH.setFormatter(formatter)
logger = logging.getLogger('views.py')
#logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)
logger.addHandler(CH)
#end ctong

NUM_TWEETS = 20 # total tweets to display
NUM_LINKS = 3 # total top links to display
NUM_LOCATIONS = 500# total locations to display on map
URL_REGEX = re.compile("http\:\/\/\S+")
CACHE_SECONDS = 864000 # 10 days

# Dicts needed for datetime calculation and convertion.
tdelta =          { 120: timedelta(minutes=2),
                    600: timedelta(minutes=10),
                    3600: timedelta(hours=1),
                    86400: timedelta(days=1)}

window =          { 120: timedelta(days=1),
                    600: timedelta(days=5),
                    3600: timedelta(days=30),
                    86400: timedelta(days=3652)} # roughly 10 years

stf =             { 120: {"date": ('%Y,%m,%d,%H,%M'),"d": 'new Date(%Y,%m-1,%d,%H,%M)'},
                    600: {"date": ('%Y,%m,%d,%H,%M'),"d": 'new Date(%Y,%m-1,%d,%H,%M)'},
                    3600: {"date": ('%Y,%m,%d,%H'),  "d": 'new Date(%Y,%m-1,%d,%H)'},
                    86400: {"date": ('%Y,%m,%d'),    "d": 'new Date(%Y,%m-1,%d)'}}
MAX_RES = max(tdelta.keys())
interval_list = sorted(tdelta.keys(), reverse=True)

def select_data(column):
    if column != 'created_at' and column != 'tweet_date':
        return
    if settings.DATABASES['default']['ENGINE'] == 'django.db.backends.postgresql_psycopg2':
        select =  { 120: {"d":    "to_char(" + column + ", 'ne\"w\" \"D\"ate(YYYY,MM-1,DD, HH24,MI)')",
                          "date": "to_char(" + column + ", 'YYYY,MM,DD,HH24,MI')"},
                    600: {"d":    "to_char(" + column + ", 'ne\"w\" \"D\"ate(YYYY,MM-1,DD, HH24,MI)')",
                          "date": "to_char(" + column + ", 'YYYY,MM,DD,HH24,MI')"},
                    3600: {"d":    "to_char(" + column + ", 'ne\"w\" \"D\"ate(YYYY,MM-1,DD,HH24)')",
                           "date": "to_char(" + column + ", 'YYYY,MM,DD,HH24')"},
                    86400: {"d":    "to_char(" + column + ", 'ne\"w\" \"D\"ate(YYYY,MM-1,DD)')",
                            "date": "to_char(" + column + ", 'YYYY,MM,DD')"}}
    else:
        select =  { 120: {"d":    "strftime('new Date(%%Y,%%m-1,%%d,%%H,%%M)', " + column + ")",
                          "date": "strftime(('%%Y,%%m,%%d,%%H,%%M') , " + column + ")"},
                    600: {"d":    "strftime('new Date(%%Y,%%m-1,%%d,%%H,%%M)', " + column + ")",
                          "date": "strftime(('%%Y,%%m,%%d,%%H,%%M') , " + column + ")"},
                    3600: {"d":    "strftime('new Date(%%Y,%%m-1,%%d,%%H)', " + column + ")",
                           "date": "strftime(('%%Y,%%m,%%d,%%H') , " + column + ")"},
                    86400: {"d":    "strftime('new Date(%%Y,%%m-1,%%d)', " + column + ")",
                            "date": "strftime(('%%Y,%%m,%%d') , " + column + ")"}}
    if column == 'tweet_date':
        for i in interval_list:
            select[i]["num_tweets"] = "num_tweets/" + str(i/60)
    
    return select

def dt2ts(dt):
    return int(time.mktime(dt.timetuple()))

def floor_date(date, tdelta):
    total_sec = tdelta.total_seconds()
    if total_sec >= 60:
        date = date.replace(microsecond=0, second=0)
    if total_sec >= 60 * 60:
        date = date.replace(minute=0)
    if total_sec >= 24 * 60 * 60:
        date = date.replace(hour=0)
    return date

def twitinfo(request):
    featured = Event.objects.filter(featured = True)
    return render_to_response('twitinfo/twitinfo.html', {"featured":featured})
       
def search_results(request):
    search = Event.normalize_name(request.GET['query'])
    events = Event.objects.filter(name = search)
    events_from_keywords = Event.objects.filter(keywords__key_word=search)
    total_events=itertools.chain(events,events_from_keywords)
    total_events=list(total_events)
    if len(total_events)==0:
        return render_to_response('twitinfo/results.html', {'error':'Sorry, the keyword you searched for does not exist.'},
                                 context_instance=RequestContext(request))
    else:
        return render_to_response('twitinfo/results.html', {'events':total_events},
                               context_instance=RequestContext(request))
                                   
def event_details(request,event_id):
    try:
       keys=[]
       event = Event.objects.get(pk=event_id)
       keywords = Keyword.objects.filter(event=event_id).values_list('key_word', flat=True)
       keys=", ".join(keywords)
    except Event.DoesNotExist:
        return render_to_response('twitinfo/details.html', {'error':'Event does not exit!'},
                                 context_instance=RequestContext(request))
  
    return render_to_response('twitinfo/details.html', {'event':event,'keywords':keys},
                                  context_instance=RequestContext(request))

class TweetDateForm(forms.Form):
    start_date = forms.DateTimeField(input_formats=["%Y-%m-%d %H:%M"],required=False)
    end_date = forms.DateTimeField(input_formats=["%Y-%m-%d %H:%M"],required=False)
    words = forms.CharField(required=False)

def display_tweets(request,event_id):
    form = TweetDateForm(request.REQUEST)
    if not form.is_valid():
        raise Http404
    start_date = form.cleaned_data['start_date']
    end_date = form.cleaned_data['end_date']
    words = form.cleaned_data['words']
    resp_string = display_tweets_impl(request, event_id, start_date, end_date, words)
    return HttpResponse(resp_string)

def display_tweets_impl(request, event_id, start_date, end_date, words):
    try:
       event = Event.objects.get(pk=event_id)
    except Event.DoesNotExist:
        return render_to_response('twitinfo/display_tweets.html', {'error':'Event does not exit!'},
                                 context_instance=RequestContext(request))

    start_date = start_date if start_date != None else event.start_date
    end_date = end_date if end_date != None else event.end_date

    key = None
    if ((start_date != None and end_date != None) or
        len(words) == 0):
        key = "tweets" + event_id
        if start_date != None and end_date != None:
            key += str(start_date)
            key += str(end_date)
            key = "".join(key.split())
        if len(words) > 0:
            key += words
        resp_string = cache.get(key)
        if resp_string != None:
            return resp_string
    
    tweets = Tweet.objects.filter(keyword__event=event_id)

    if start_date != None:
        tweets = tweets.filter(created_at__gte = start_date)
    if end_date != None:
        tweets = tweets.filter(created_at__lte = end_date)
    tweets = tweets.order_by("created_at")#+

    if len(words) == 0:
        tweets = tweets[:NUM_TWEETS]
    else:
        words = words.split(",")
        matched_tweets = []
        already_tweets = set()
        for tweet in tweets[:500]:
            count = 0
            text = tweet.tweet.lower()
            if "rt" in text:
                count -= 2
            text = URL_REGEX.sub("WEBSITE", text)
            if text not in already_tweets:
                for word in words:
                    if word in text:
                        count += 1
                matched_tweets.append((tweet, count))
                already_tweets.add(text)
        matched_tweets.sort(cmp=lambda a,b: cmp(b[1],a[1]))
        tweets = [t[0] for t in matched_tweets[:min(NUM_TWEETS,len(matched_tweets))]]

    t = loader.get_template('twitinfo/display_tweets.html')
    resp_string = t.render(Context({'tweets': tweets,'event':event}))

    if key != None and (len(tweets) == NUM_TWEETS or is_event_finalized_as_of(event_id, end_date)):
        cache.set(key, resp_string, CACHE_SECONDS)
    
    return resp_string

def display_links(request,event_id):
    form = TweetDateForm(request.REQUEST)
    if not form.is_valid():
        raise Http404
    start_date = form.cleaned_data['start_date']
    end_date = form.cleaned_data['end_date']
    try:
        resp_string = display_links_impl(request, event_id, start_date, end_date)
    except:
        traceback.print_exc()
    return HttpResponse(resp_string)

def display_links_impl(request, event_id, start_date, end_date):
    try:
       event = Event.objects.get(pk=event_id)
    except Event.DoesNotExist:
        return 'Event does not exit!'

    start_date = start_date if start_date != None else event.start_date
    end_date = end_date if end_date != None else event.end_date

    key = None
    if ((start_date != None and end_date != None) or
        (start_date == None and end_date == None)):
        key = "links" + event_id
        if start_date != None and end_date != None:
            key += str(start_date)
            key += str(end_date)
            key = "".join(key.split())
        resp_string = cache.get(key)
        if resp_string != None:
            return resp_string

    tweets = Tweet.objects.filter(keyword__event=event_id)

    if start_date != None:
        tweets = tweets.filter(created_at__gte = start_date)
    if end_date != None:
        tweets = tweets.filter(created_at__lte = end_date)
    tweets = tweets.order_by("created_at")#+
    links = {}

    count_tweets = 0
    for tweet in tweets[:500]:
        text = tweet.tweet
        incr = 1
        if "RT" in text:
            incr = .5
        for match in URL_REGEX.findall(text):
            count = links.get(match, 0.0)
            count += incr
            links[match] = count
        count_tweets += 1

    linkcounts = links.items()
    linkcounts.sort(key = itemgetter(1), reverse = True)
    displaylinks = []
    for i in range(0, min(len(linkcounts), NUM_LINKS)):
        if linkcounts[i][1] > 2.5:
            displaylinks.append((linkcounts[i][0], int(linkcounts[i][1])))
    t = loader.get_template('twitinfo/display_links.html')
    resp_string = t.render(Context({'links': displaylinks}))

    if key != None and (count_tweets == 500 or is_event_finalized_as_of(event_id, end_date)):
        cache.set(key, resp_string, CACHE_SECONDS)
    return resp_string
    
class EventForm(forms.Form):
    title=forms.CharField(max_length=100)
    key_words = forms.CharField()
    start_date = forms.DateTimeField(input_formats=["%Y-%m-%d %H:%M"],required=False)
    end_date = forms.DateTimeField(input_formats=["%Y-%m-%d %H:%M"],required=False)
    parent_id = forms.IntegerField(widget=forms.HiddenInput,required=False)
    

def create_event(request):
   if request.method == 'POST': # If the form has been submitted...
       form = EventForm(request.POST) # A form bound to the POST data
       if form.is_valid():
           name = form.cleaned_data['title']
           name = Event.normalize_name(name)
           key_words = form.cleaned_data['key_words']
           list_keywords = Keyword.normalize_keywords(key_words)
           keyobjs=[]
           for key in list_keywords:
               try:
                   fkeyword = Keyword.objects.get(key_word = key)
               except Keyword.DoesNotExist:
                   fkeyword = Keyword(key_word = key)
                   fkeyword.save() 
               keyobjs.append(fkeyword)  
           
           e = Event(name = name,start_date = None,end_date = None)
           try:
               e.start_date = form.cleaned_data['start_date']
           except:
               pass
           try:
               e.end_date = form.cleaned_data['end_date']
           except:
               pass
           e.save()
           e.keywords = keyobjs
           try:
               parent = form.cleaned_data['parent_id']
               parent_event = Event.objects.get(id=parent)
               parent_event.children.add(e)
               cache.delete("graph" + str(parent)) # clear parent view to include child
           except Event.DoesNotExist:
               pass
           #ctong: workaround
           #return HttpResponseRedirect('detail/%d' % (e.id)) # Redirect after POST
           return HttpResponseRedirect('/detail/%d' % (e.id)) # Redirect after POST
   else:
       # initialize the form with a set of values that are passed in as data. If there are no initial values,initialize an empty form.
       try:
           parent_id=request.GET["parent_id"]
           keywords=request.GET["keywords"]
           sd=request.GET["start_date"]
           ed=request.GET["end_date"]
           data={'start_date':sd,'end_date':ed,'key_words':keywords,'parent_id':parent_id,'title':" "}
           form = EventForm(data) 
       except:
           form = EventForm()
   return render_to_response('twitinfo/create_event.html', {'form': form}, context_instance=RequestContext(request))    
   

def find_end_dates(tweets,list_peaks):
    i=0
    k=0
    
    if len(list_peaks) > 0:
        while(i<len(list_peaks) and i+1<len(list_peaks)):
            for j in range(len(tweets)):
                if(list_peaks[i]["start_date"]==tweets[j]['date']):
                    k=j+1
                    break
            while(k<len(tweets)):
                    if(list_peaks[i+1]['start_date']==tweets[k]['date'] or tweets[k]['num_tweets']<=list_peaks[i]["start_freq"] or k==len(tweets)-1):
                        end_date=tweets[k]['date']
                        list_peaks[i]["end_date"]=end_date
                        break
                    k+=1
            i+=1
        for l in range(len(tweets)):
                if(list_peaks[len(list_peaks)-1]["start_date"]==tweets[l]['date']):
                    k=l+1
                    break
                    
        while(k<len(tweets)):
                    if( tweets[k]['num_tweets']<=list_peaks[len(list_peaks)-1]["start_freq"] or k==(len(tweets)-1)):
                        end_date=tweets[k]['date']
                        list_peaks[len(list_peaks)-1]["end_date"]=end_date
                    k+=1
    return list_peaks  

def words_by_tfidf(dic, keywords):
    logger.debug('Entering words_by_tfidf')
    freq_words = []
##        freq_words = WordFrequency.objects.filter(word__in = list_keys).values_list('word', 'idf')

    temp_words = WordFrequency.objects.all().values_list('word', 'idf')
    for word, idf in temp_words:
        if dic.has_key(word):
            freq_words.append([word, idf])

    logger.debug('words_by_tfidf returns %s frequent words among %s words', len(freq_words), len(dic))

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

    logger.debug('Exiting words_by_tfidf')
    return words

def find_max_terms(tweets, keywords):
    logger.debug('Entering find_max_terms')
    total_freq=0
    words_dict={}
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stopwords.add("rt")

    logger.debug('Retrieved %s stopwords', len(stopwords))

    for tweet in tweets:    
        text = tweet.tweet
        tweet_text=text.lower().replace("'","").split()      
        for word in tweet_text:
            if word in stopwords:
                continue
            if words_dict.has_key(word):
                words_dict[word]+=1
            else:
                words_dict[word]=1

    logger.debug('Populated tweet dictionary of %s words', len(words_dict))
    return words_by_tfidf(words_dict, keywords)
 
def annotate_peaks(peaks,tweets,event_id,keywords):
    logger.debug('Entering annotate_peaks')
    list_keywords = ", ".join(keywords)
    for peak in peaks:
        sdt = convert_date(peak['start_date'])
        edt = convert_date(peak['end_date'])
        t=Tweet.objects.filter(keyword__event=event_id).filter(created_at__gte=sdt).filter(created_at__lte=edt)
        logger.debug('Retrieved %s tweets for peak (%s, %s)', len(t), sdt, edt)
        sorted_list=find_max_terms(t, keywords)

        peak_tweet = peak['peak_tweet']
        peak_tweet['title']="'" +", ".join(sorted_list[:5])+"'"
        peak_tweet['data']={'event':event_id,'keywords':list_keywords,'start_date':sdt.strftime("%Y-%m-%d %H:%M"),'end_date':edt.strftime("%Y-%m-%d %H:%M")}
    return tweets

def convert_date(date):
    d=date.split(',')
    d=map(int , d)
    dt=datetime(*d)
    return dt

def pad_tweets_gap(tweets, interval):
    tweets[0]['title'] = 'null'
    tweets[0]['data'] = 'null'
    tweets[0]['children'] = None
    i=1
    while i < len(tweets):
        tweets[i]['children'] = None
        if not tweets[i].has_key('data'):
            tweets[i]['title'] = 'null'
            tweets[i]['data'] = 'null'
        sdt_p = convert_date(tweets[i-1]['date'])
        sdt_n = convert_date(tweets[i]['date'])
        delta_d = (sdt_n-sdt_p).total_seconds()/interval

        if delta_d > 1:
            j=0
            while(j<delta_d-1):
                insert_tweet = {'title':'null','num_tweets':0,'data':'null','children':None}
                sdt_p = sdt_p + tdelta[interval]    
                insert_tweet['date'] = sdt_p.strftime(stf[interval]['date'])
                tweets.insert(i+j, insert_tweet)
                j += 1
            i += j
        i += 1

class GraphForm(forms.Form):
    #start_date = forms.DateTimeField(input_formats=["%Y-%m-%d %H:%M"],required=False)
    #end_date = forms.DateTimeField(input_formats=["%Y-%m-%d %H:%M"])
    end_date = forms.IntegerField() # epoch time
    interval = forms.IntegerField()

    def is_valid(self):
        valid = forms.Form.is_valid(self)
        return valid and (self.cleaned_data['interval'] in interval_list)

def create_graph(request,event_id):
    key = "graph" + event_id
    resp_string = cache.get(key)
    logger.info('Entering create_graph')
    if resp_string == None:
        try:
            resp_string = create_graph_impl2(request, event_id)
        except:
            traceback.print_exc()
            raise
        # ctong: do not blindly cache response because there can be new data between requests
        # cache.set(key, resp_string, CACHE_SECONDS)
    resp_string = request.GET["jsoncallback"] + resp_string
    return HttpResponse(resp_string)

def create_graph_impl2(request, event_id):
    logger.info('Entering create_graph_impl')
    form = GraphForm(request.GET)
    if not form.is_valid():
        raise Http404
    request_edate = datetime.fromtimestamp(form.cleaned_data['end_date'])
    interval = form.cleaned_data['interval']
    logger.debug('Form params end_date: %s. interval: %d', request_edate, interval)

    e = Event.objects.get(id = event_id)
    logger.debug('User specified event date range [%s, %s]', e.start_date, e.end_date)
    tweets = Tweet.objects.filter(keyword__event = event_id)
    keywords = Keyword.objects.filter(event__id = event_id)

    key_first_date = 'event' + event_id + 'first_date'
    first_date = cache.get(key_first_date)
    if first_date == None:
        first_date = tweets.order_by('created_at')[0].created_at
        if first_date == None:
            logger.debug('No tweets for this event yet')
            return
        cache.set(key_first_date, first_date, CACHE_SECONDS)

    last_tweet_id = max([kw.max_indexed for kw in keywords])
    if last_tweet_id < 0:
        logger.debug('No tweets for this event yet')
        return
    last_tweet = Tweet.objects.filter(id = last_tweet_id).values()[0]
    last_date = last_tweet['created_at']
    logger.debug('Actual tweets date range: [%s, %s]', first_date, last_date)

    extra = {'final':'false'}
    last_tweet['created_at'] = dt2ts(last_tweet['created_at'])
    extra['latest_tweet'] = last_tweet
    if e.end_date:
        edate = min(e.end_date, last_date, request_edate)
        if e.end_date <= last_date:
            extra['final'] = 'true'
    else:
        edate = min(last_date, request_edate)
    if e.start_date:
        sdate = max(e.start_date, first_date, edate - window[interval])
    else:
        sdate = max(first_date, edate - window[interval])
    logger.debug('Result date range for interval %s: [%s, %s]', interval, sdate, edate)
    total_sec = (edate-sdate).total_seconds()

##    # Determine boundary of windows, starting with leftmost window (lowest resolution)
##    last_stats = dict()
##    wdate = {MAX_RES+1 : sdate} # upper boundary of windows, i.e right side
##    for res in res_list:
##        try:
##            last_stats[res] = EventStats.objects.get(event__id = event_id, resolution = res, pos = -1)
##        except:
##            pass
##        if last_stats.has_key(res):
##            wdate[res] = min(last_stats[res].tweet_date, max(edate - window[res], wdate[res+1]))
##        else:
##            wdate[res] = wdate[res+1]
##    logger.debug(last_stats)
##    logger.debug(wdate)

    list_keywords = ", ".join([kw.key_word for kw in keywords])
        
##    # Retrieve tweet freqs and annotate peaks in each window
##    tweetfreqs_dict = dict()
##    for res in res_list:
##        tweetfreqs = TweetFrequency.objects.filter(event__id = event_id).filter(tweet_date__gt = wdate[res+1]).filter(tweet_date__lte = wdate[res]).filter(resolution = res)
##        tweetfreqs = tweetfreqs.extra(select = select_data('tweet_date')[res]).values('d','date','num_tweets').order_by('tweet_date')
##        tweetfreqs = list(tweetfreqs)
##        logger.debug('DB retrieves %s tweet freqs with resolution %s', len(tweetfreqs), res)

##        if len(tweetfreqs) == 0:
##            tweetfreqs_dict[res] = tweetfreqs
##            continue

##        peaks = Peak.objects.filter(event__id = event_id).filter(start_date__gt = wdate[res+1]).filter(start_date__lte = wdate[res]).order_by('start_date')
##        peaks = list(peaks)
##        logger.debug('DB retrieves %s peaks with resolution %s', len(peaks), res)

##        peaks_dict = dict()
##        for peak in peaks:
##            peaks_dict[peak.peak_date] = peak

##        for tweetfreq in tweetfreqs:
##            peak_date = convert_date(tweetfreq['date'])
##            if peaks_dict.has_key(peak_date):
##                peak = peaks_dict[peak_date]
##                tweetfreq['title'] = peak.freqwords
##                tweetfreq['data'] = {'event':event_id,'keywords':list_keywords,'start_date':peak.start_date.strftime("%Y-%m-%d %H:%M"),'end_date':peak.end_date.strftime("%Y-%m-%d %H:%M")}

##        pad_tweets_gap(tweetfreqs, res)
##        tweetfreqs_dict[res] = tweetfreqs
##        logger.debug('Resolution %s has %s tweet freqs', res, len(tweetfreqs))

##    tweets = tweets.filter(keyword__event = event_id).filter(created_at__gt = wdate[1]).filter(created_at__lte = edate).extra(select = select_data('created_at')[res]).values('d','date').annotate(num_tweets = Count('tweet')).order_by('date')
##    tweets = list(tweets)
##    logger.debug('DB retrieves %s unprocessed tweet freqs', len(tweets))
##    pad_tweets_gap(tweets, min(res_list))
##    logger.debug('%s unprocessed tweet freqs', len(tweets))
##    tweetfreqs_dict[0] = tweets

##    tweetfreqs_list = []
##    # Combine tweet freqs from all windows into a single list
##    prev_res = res_list[0]
##    for i in res_list + [0]:
##        tweetfreqs = tweetfreqs_dict[i]
##        res = max(i, 1)
##        if len(tweetfreqs):
##            if len(tweetfreqs_list):
##                for pad_res in [prev_res, res]:
##                    gap = [tweetfreqs_list[-1], tweetfreqs[0]]
##                    pad_tweets_gap(gap, pad_res)
##                    gap = gap[1:-1]
##                    logger.debug('Res: %s, padding tweet freqs with %s blank data of resolution %s', res, len(gap), pad_res)
##                    tweetfreqs_list.extend(gap)
##            tweetfreqs_list.extend(tweetfreqs)
##            prev_res = res
##    logger.debug('Final result: %s tweet freqs', len(tweetfreqs_list))

## new code

    tweetfreqs = TweetFrequency.objects.filter(event__id = event_id).filter(tweet_date__gte = sdate).filter(tweet_date__lte = edate).filter(interval = interval)
    tweetfreqs = tweetfreqs.extra(select = select_data('tweet_date')[interval]).values('date','num_tweets').order_by('tweet_date')
    tweetfreqs = list(tweetfreqs)
    logger.debug('DB retrieves %s tweet freqs with interval %s', len(tweetfreqs), interval)
    
    if len(tweetfreqs) == 0:
        logger.error('Expecting tweet freqs but missing')
        return
    
    peaks = Peak.objects.filter(event__id = event_id).filter(start_date__gte = sdate).filter(start_date__lte = edate).filter(interval = interval).order_by('start_date')
    peaks = list(peaks)
    logger.debug('DB retrieves %s peaks with interval %s', len(peaks), interval)

    peaks_dict = dict()
    for peak in peaks:
        peaks_dict[peak.peak_date] = peak

    for tweetfreq in tweetfreqs:
        peak_date = convert_date(tweetfreq['date'])
        if peaks_dict.has_key(peak_date):
            peak = peaks_dict[peak_date]
            tweetfreq['title'] = peak.freqwords
            tweetfreq['data'] = {'event':event_id,'keywords':list_keywords,'start_date':peak.start_date.strftime("%Y-%m-%d %H:%M"),'end_date':peak.end_date.strftime("%Y-%m-%d %H:%M")}

    pad_tweets_gap(tweetfreqs, interval)
    logger.debug('Interval %s has %s tweet freqs', interval, len(tweetfreqs))
    tweetfreqs_list = tweetfreqs

    for tweetfreq in tweetfreqs_list:
        tweetfreq['d'] = dt2ts(convert_date(tweetfreq['date']))

    env = Environment(loader=FileSystemLoader(settings.TEMPLATE_DIRS))
    t = env.get_template('twitinfo/create_graph2.html')
    resp_string = t.render(tweets=tweetfreqs_list, extra=extra)

    logger.info('Exiting create_graph_impl')
    return resp_string

def create_graph_impl(request, event_id):
    #ctong
    logger.info('Entering create_graph_impl')
    e = Event.objects.get(id=event_id)
    sdate = e.start_date
    edate = e.end_date
    tweets = Tweet.objects.filter(keyword__event = event_id)
    
    if sdate == None:
        sdate=tweets.order_by('created_at')[0].created_at
    if edate == None:
        edate=tweets.order_by('-created_at')[0].created_at
        
    tdelta=(edate-sdate)
    total_sec=tdelta.seconds + tdelta.days * 24 *3600
    total_min=total_sec / 60.0
    total_hours=total_min / 60.0
    
    if total_min <= 1440:
        td=timedelta(minutes=1)
        sec_divisor = 60
        stf = {"date": ('%Y,%m,%d,%H,%M'),"d": 'new Date(%Y,%m-1,%d,%H,%M)'}
        if settings.DATABASES['default']['ENGINE'] == 'postgresql_psycopg2':
            select_data = {"d": "to_char(created_at, 'ne\"w\" \"D\"ate(YYYY,MM-1,DD, HH24,MI)')" , "date":"to_char(created_at, 'YYYY,MM,DD,HH24,MI')"}
        else:
            select_data = {"d": "strftime('new Date(%%Y,%%m-1,%%d,%%H,%%M)', created_at)" , "date":"strftime(('%%Y,%%m,%%d,%%H,%%M') , created_at)"}
      
    elif total_hours <= 2016: # 24 hours x 28 days x 3 = about 3 months
        td=timedelta(hours=1)
        sec_divisor = 3600
        stf = {"date": ('%Y,%m,%d,%H'),"d": 'new Date(%Y,%m-1,%d,%H)'}
        if settings.DATABASES['default']['ENGINE'] == 'postgresql_psycopg2':
            select_data = {"d": "to_char(created_at, 'ne\"w\" \"D\"ate(YYYY,MM-1,DD,HH24)')" , "date":"to_char(created_at, 'YYYY,MM,DD,HH24')"}
        else:
            select_data = {"d": "strftime('new Date(%%Y,%%m-1,%%d,%%H)', created_at)" , "date":"strftime(('%%Y,%%m,%%d,%%H') , created_at)"}
    else:
        td=timedelta(days=1)
        sec_divisor = 86400
        stf = {"date": ('%Y,%m,%d'),"d": 'new Date(%Y,%m-1,%d)'}
        if settings.DATABASES['default']['ENGINE'] == 'postgresql_psycopg2':
            select_data = {"d": "to_char(created_at, 'ne\"w\" \"D\"ate(YYYY,MM-1,DD)')" , "date":"to_char(created_at, 'YYYY,MM,DD')"}
        else:
            select_data = {"d": "strftime('new Date(%%Y,%%m-1,%%d)', created_at)" , "date":"strftime(('%%Y,%%m,%%d') , created_at)"}
    
    tweets = tweets.filter(created_at__gte = sdate).filter(created_at__lte = edate).extra(select = select_data).values('d','date').annotate(num_tweets = Count('tweet')).order_by('date')
    tweets=list(tweets)

    logger.debug('DB retrieves %s tweet groups: %s', len(tweets), tweets)

    i = 1
    detector = MeanOutliers.factory()
    list_peaks = []
    # loop through the tweets and detect a peak based on mean deviation function provided. save the start date
    # and the date of the peak in a dictionary.  save each peak in list_peaks.
    while i < len(tweets):
        tweets[0]['title'] = 'null'
        tweets[0]['data'] = 'null'
        # sd_p=tweets[i-1]['date'].split(',')
        # sd_p=map(int , sd_p)
        # sdt_p=datetime(*sd_p)
        sdt_p=convert_date(tweets[i-1]['date'])
        sd_n=tweets[i]['date'].split(',')
        sd_n=map(int , sd_n)
        sdt_n=datetime(*sd_n)
        delta_d=(sdt_n-sdt_p)
        delta_d = (delta_d.seconds + delta_d.days * 24 *3600)/sec_divisor  

        count=0
        if delta_d != 1:
            j=0
            while(j<delta_d-1):
                insert_tweet={'title':'null','num_tweets':0,'data':'null','children':'null'}
                sdt_p = sdt_p+td    
                insert_tweet['date']=sdt_p.strftime(stf['date'])
                insert_tweet['d']=sdt_p.strftime(stf['d'])
                tweets.insert(i+j,insert_tweet)
                j+=1

        current_val = tweets[i]['num_tweets']
        previous_val = tweets[i-1]['num_tweets']
        mdiv = detector(None,tweets[i]['num_tweets'], 1)
        if mdiv > 2.0 and current_val > previous_val and current_val > 10:
            start_freq = previous_val 
            start_date = tweets[i-1]['date']
            # once a peak is detected, keep climbing up the peak until the maximum is reached. store the peak date and keep
            # running the mdiv function on each value because it is requires previous values to calculate the mean.
            while(current_val > previous_val):
                  tweets[i]['title'] = 'null'
                  tweets[i]['data'] = 'null'
                  if i+1<len(tweets):
                      i+=1
                      mdiv = detector(None,tweets[i]['num_tweets'], 1)
                      current_val = tweets[i]['num_tweets']
                      previous_val = tweets[i-1]['num_tweets'] 
                      peak_date = tweets[i-1]['date']
                      peak_tweet = tweets[i-1]
                  else:
                      peak_date = tweets[i]['date']
                      peak_tweet = tweets[i]
                      i+=1
                      break
            #d = {"start_date":start_date,"start_freq":start_freq ,"peak_date":peak_date}
            d = {"start_date":start_date,"start_freq":start_freq ,"peak_date":peak_date, "peak_tweet":peak_tweet}
            list_peaks.append(d)
        else:
            tweets[i]['title'] = 'null'
            tweets[i]['data'] = 'null'
            i+=1

    logger.debug('MeanOutliers returns %s peak tweet groups: %s', len(list_peaks), list_peaks)
    logger.debug('Tweets: %s', tweets)
    
    keywords = Keyword.objects.filter(event__id = event_id)
    words = [kw.key_word for kw in keywords]
    peaks = find_end_dates(tweets,list_peaks)

    logger.debug('find_end_dates returns %s peak tweet groups: %s', len(peaks), peaks)

    try:
        tweets = annotate_peaks(peaks,tweets,event_id,words)
    except:
        traceback.print_exc()

    logger.debug('annotate_peaks returns %s tweets: %s', len(tweets), tweets)
    logger.debug('annotate_peaks returns %s peak tweet groups: %s', len(peaks), peaks)

    try:
        children = e.children.order_by('start_date')
        tweets = peak_child_detection(children,list_peaks,tweets)
    except:
        tweets = tweets

    logger.debug('peak_child_detection returns %s tweets: %s', len(tweets), tweets)
    logger.debug('peak_child_detection returns %s peak tweet groups: %s', len(list_peaks), list_peaks)
    
    t = loader.get_template('twitinfo/create_graph.html')
    resp_string = t.render(Context({ 'tweets': tweets }))

    logger.info('Exiting create_graph_impl')
    return resp_string
 
def peak_child_detection(children,list_peaks,tweets):
    list_overlaps=[]
    i=0
    savej=0
    count=0
    while(i<len(children)):
        j=savej
        while(j<len(list_peaks)):
            
            if ((children[i].start_date >= convert_date(list_peaks[j]['start_date']) and children[i].start_date <= convert_date(list_peaks[j]['end_date'])) or
                (children[i].end_date >=convert_date(list_peaks[j]['start_date'])  and children[i].end_date<=convert_date(list_peaks[j]['end_date'])) or
                    (convert_date(list_peaks[j]['start_date'])>=children[i].start_date and convert_date(list_peaks[j]['start_date'])<=children[i].end_date) or
                    (convert_date(list_peaks[j]['end_date']) >= children[i].start_date and convert_date(list_peaks[j]['end_date'])<=children[i].end_date)):
                   
                    if(count==1):
                        savej=j-1
                    
                    if list_peaks[j].has_key("children"):   
                        list_peaks[j]["children"].append(children[i])
                    else:
                        list_peaks[j]["children"]=[children[i]] 
                    
                    j+=1
                
            elif(convert_date(list_peaks[j]['start_date']) > children[i].end_date):
                j=savej
                break
            elif(convert_date(list_peaks[j]['end_date']) < children[i].start_date):
                count+=1
                j+=1
            
        i+=1
       
    j=0
    for peak in list_peaks:
        if peak.has_key("children"):
            while(j<len(tweets)):
                if convert_date(tweets[j]['date'])==convert_date(peak["peak_date"]):
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

def is_event_finalized_as_of(event_id, date):
    # an event is finalized as of a date if all tweets prior to that date have been peak-detected
    if date == None:
        return False
    key_last_tweet = 'event' + str(event_id)
    key_last_tweet += 'interval' + str(min(interval_list)) + 'last_tweet' 
    prev_last_tweet = cache.get(key_last_tweet)
    if prev_last_tweet == None:
        return False
    if date >= prev_last_tweet['created_at']:
        return False
    return True

def create_pieChart(request, event_id):
    form = TweetDateForm(request.REQUEST)
    if form.is_valid():
        start_date = form.cleaned_data['start_date']
        end_date = form.cleaned_data['end_date']
    try:
        resp_string = create_pieChart_impl(request, event_id, start_date, end_date)
    except:
        traceback.print_exc()
    resp_string = request.GET["jsoncallback"] + resp_string
    return HttpResponse(resp_string)

def create_pieChart_impl(request, event_id, start_date, end_date):
    event = Event.objects.get(pk=event_id)
    start_date = start_date if start_date != None else event.start_date
    end_date = end_date if end_date != None else event.end_date

    key = None
    if start_date != None and end_date != None:
        key = "pie" + event_id 
        key += str(start_date)
        key += str(end_date)
        key = "".join(key.split())
        resp_string = cache.get(key)
        if resp_string != None:
            return resp_string

    tweets = TweetFrequency.objects.filter(event__id = event_id, interval = min(interval_list))
    if start_date != None:
        tweets = tweets.filter(tweet_date__gte = start_date)
    if end_date != None:
        tweets = tweets.filter(tweet_date__lte = end_date)

    tweets = tweets.aggregate(sum_positive = Sum('pos_sentiment'), sum_negative = Sum('neg_sentiment'))
    sum_positive = round(tweets['sum_positive'] if tweets['sum_positive'] != None else 0)
    sum_negative = round(tweets['sum_negative'] if tweets['sum_negative'] != None else 0)* -1.0

    rows = []
    cols = [{'id': 'sentiment','label':'SENTIMENT','type': 'string'},{'id': 'frequency','label':'FREQUENCY' ,'type': 'number'}]
 
    p ='positive'
    rows.append( {'c':[ {'v':p},{'v':sum_positive} ] } )
    p='negative'
    rows.append( {'c':[ {'v':p},{'v':sum_negative} ] } )
      
    data={'cols':cols,'rows':rows}
    resp_string = "("+json.dumps(data)+");"

    if key != None and is_event_finalized_as_of(event_id, end_date):
        cache.set(key, resp_string, CACHE_SECONDS)

    return resp_string

def create_map(request, event_id):
    form = TweetDateForm(request.REQUEST)
    if not form.is_valid():
        raise Http404
    start_date = form.cleaned_data['start_date']
    end_date = form.cleaned_data['end_date']
    resp_string = create_map_impl(request, event_id, start_date, end_date)
    resp_string = request.GET["jsoncallback"] + resp_string
    return HttpResponse(resp_string)

def create_map_impl(request, event_id, start_date, end_date):
    event = Event.objects.get(pk=event_id)
    start_date = start_date if start_date != None else event.start_date
    end_date = end_date if end_date != None else event.end_date

    key = None
    if ((start_date != None and end_date != None) or
        (start_date == None and end_date == None)):
        key = "map" + event_id
        if start_date != None and end_date != None:
            key += str(start_date)
            key += str(end_date)
            key = "".join(key.split())
        resp_string = cache.get(key)
        if resp_string != None:
            return resp_string

    tweets = Tweet.objects.filter(keyword__event=event)
    tweets = tweets.values('tweet', 'latitude', 'longitude', 'profile_image_url', 'sentiment')
    
    if start_date != None:
        tweets = tweets.filter(created_at__gte = start_date)
    if end_date != None:
        tweets = tweets.filter(created_at__lte = end_date)
    tweets = tweets.filter(latitude__isnull = False)
    tweets = tweets.order_by("created_at")#+
    data = []

    count = 0
    for tweet in tweets[:NUM_LOCATIONS]:
        # perturb locations slightly to avoid two tweets in "New York"
        # occluding one-anotheir on the map
        tweet['latitude'] += random.uniform(-.012, .012)
        tweet['longitude'] += random.uniform(-.012, .012)
        data.append({'text': tweet['tweet'],
                     'latitude': tweet['latitude'],
                     'longitude': tweet['longitude'],
                     'image': tweet['profile_image_url'],
                     'sentiment': tweet['sentiment']})
        count += 1
    
    resp_string = "("+json.dumps(data)+");"

    if key != None and (count == NUM_LOCATIONS or is_event_finalized_as_of(event_id, end_date)):
        cache.set(key, resp_string, CACHE_SECONDS)

    return resp_string
