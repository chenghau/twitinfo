google.load('visualization', '1', {'packages':['annotatedtimeline']});
google.load('visualization', '1', {'packages':["corechart"]});
var old_infowindow = null;
var graph;
var graph1;
var graph2;
var graph_mode = 'hist';
var graph_data;
var event_id;
var interval;
var prev_interval;
var last_frequency;
var prev_last_frequency = -1;
var no_more_data = false;

var interval_val = {};
interval_val['1d'] = 120;
interval_val['5d'] = 600;
interval_val['1m'] = 3600;
DEFAULT_INTERVAL = 86400;
REALTIME_INTERVAL = 120;

var interval_string = {};
interval_string[120] = '2 minutes';
interval_string[600] = '10 minutes';
interval_string[3600] = 'hour';
interval_string[86400] = 'day';

function get_interval(key)  
{
  if (key in interval_val)
    return interval_val[key];
  else
    return DEFAULT_INTERVAL;
}

function initialize(id) {
  event_id = id;
  interval = get_interval($('#zoom_div .options.selected').html());
  prev_interval = interval;
  init_graph();
  request_map("");
  request_graph();
  request_pie("");
  $('#zoom_div .options').click(
    function() 
    {
      $('#zoom_div .options').removeClass('selected');
      $(this).addClass('selected');
      new_interval = get_interval($(this).html());
      console.log('new interval is', new_interval);
      
      if (new_interval != interval)
      {
        console.log('updating graph from interval %s to interval %s', interval, new_interval);
        prev_interval = interval;
        interval = new_interval;
        request_graph();
      }
    });
}
function request_map(extra_args) {
  $.getJSON("/detail/create_map/" + event_id + "?jsoncallback=?&" + extra_args,
            function(data){
              draw_map(data)
                } 
    );
}
function request_graph() {
  console.log('requesting graph');
  var now = new Date();
  now = new Date(now.getTime() + now.getTimezoneOffset()*60*1000).getTime()/1000;
  if (graph_mode == 'real')
    interval = REALTIME_INTERVAL;
  
  var url = "/detail/create_graph/" + event_id + "?jsoncallback=?";
  url += "&end_date=" + now.toFixed(0);
  url += "&interval=" + interval.toString();
  $.getJSON(url,
            function(data){
              parse_graph(data);
              toggle_graph();
              
              if (graph_mode == 'hist')
              {
                draw_graph(data);
                update_frequency(data, false);
              }
              else if (graph_mode == 'real')
              {
                draw_real_graph(data);
                update_frequency(data, true);
                setTimeout('request_graph()', 5000);
              }

              if (!no_more_data)
                setTimeout('request_graph()', 5000);
            } 
    );
}
function request_pie(extra_args) {
  $.getJSON("/detail/create_piChart/" + event_id + "?jsoncallback=?&" + extra_args,
            function(data){
              draw_piChart(data)
                } 
    );
}
function draw_graph(data) {
  console.log('Drawing graph');
  graph_data = new google.visualization.DataTable(data,0.6);
  graph.draw(graph_data, {displayAnnotations:true, allowHtml:true, annotationsWidth: 30, displayAnnotationsFilter: true, displayZoomButtons: false, scaleType:'maximized'});
}
function draw_real_graph(data) {
  console.log('Drawing real graph');
  graph_data = new google.visualization.DataTable(data,0.6);
  //graph.draw(graph_data, {allowRedraw:true,displayAnnotations: false,allowHtml:true});
  graph.draw(graph_data, {displayAnnotations:true, allowHtml:true, annotationsWidth: 30, displayAnnotationsFilter: true, displayZoomButtons: false, scaleType:'maximized'});
}
function init_graph() {
  graph1 = new google.visualization.AnnotatedTimeLine(document.getElementById('graph_div1'));
  graph2 = new google.visualization.AnnotatedTimeLine(document.getElementById('graph_div2'));
  graph = graph1;
  $('graph_div1').css('visibility', 'visible');
  $('graph_div2').css('visibility', 'hidden');
  google.visualization.events.addListener(graph1, 'select', handle_select);
  google.visualization.events.addListener(graph2, 'select', handle_select);
  google.visualization.events.addListener(graph1, 'ready', handle_ready);
  google.visualization.events.addListener(graph2, 'ready', handle_ready);
}

function handle_select(event) {
  console.log('In graph select event handler');
  row = graph.getSelection()[0].row;
  create_link = graph_data.getValue(row, 3);
  arg_str = create_link.split("?")[1];
  args = arg_str.split("&");
  extra_args = args[0] + "&" + args[1];
  //extra_args = extra_args + "end_date=" + formatDate(graph_data.getValue(row,0), 'yyyy-MM-dd HH:mm');
  request_pie(extra_args);
  request_map(extra_args);
  tweet_src = "/display_links/" + event_id + "?";
  tweet_src = tweet_src + extra_args;
  $("#links_iframe").attr("src", tweet_src);
  tweet_src = "/display_tweets/" + event_id + "?";
  tweet_src = tweet_src + extra_args;
  words = graph_data.getValue(row, 2);
  words = words.replace(/ /g, "");
  tweet_src = tweet_src + "&words=" + words;
  $("#tweets_iframe").attr("src", tweet_src);
}

function handle_ready(event) 
{
  console.log('In graph ready event handler');
  if (graph == graph1)
    graph_id = '#graph_div1';
  else if (graph == graph2)
    graph_id = '#graph_div2';
  
  console.log('Showing', graph_id);
  $('.graph').css('visibility', 'hidden');
  $(graph_id).css('visibility', 'visible');
}

function draw_piChart(data) {
  var data = new google.visualization.DataTable(data,0.6);
  var chart = new google.visualization.PieChart(document.getElementById('pchart_div'));
  chart.draw(data, {displayAnnotations: true});
}
function draw_map(data) {
  var myLatlng = new google.maps.LatLng(39.095963,-97.031250);
  var myOptions = {
    zoom: 1,
    center: myLatlng,
    mapTypeId: google.maps.MapTypeId.ROADMAP
  }
  var map = new google.maps.Map(document.getElementById("map_canvas"), myOptions);
  for (var i = 0; i < data.length; i++) {
    var tweet = data[i];
    var icon = 'http://labs.google.com/ridefinder/images/mm_20_gray.png';
    if (tweet.sentiment > 0) {
      icon = 'http://labs.google.com/ridefinder/images/mm_20_blue.png';
    } else if (tweet.sentiment < 0) {
      icon = 'http://labs.google.com/ridefinder/images/mm_20_red.png';
    }
    var marker = new google.maps.Marker({
    position: new google.maps.LatLng(tweet.latitude, tweet.longitude), 
          map: map,
          title: tweet.text,
          icon: icon
          });
    attach_message(map, marker, tweet);
  }
}
            
function attach_message(map, marker, tweet) {
  content = "<div style='float: left;'><img src='" + tweet.image + "'></div>" + tweet.text;
  var infowindow = new google.maps.InfoWindow(
    { content: content }
    );
  google.maps.event.addListener(marker, 'click', function() {
                                  if (old_infowindow != null) {
                                    old_infowindow.close();
                                  }
                                  infowindow.open(map,marker);
                                  old_infowindow = infowindow;
                                });
}

function get_tweetsum() 
{
  var group = google.visualization.data.group(graph_data, [1], 
    [{ 
    // get the sum of column 1 
    'column': 1, 
    'type': 'number', 
    'aggregation': google.visualization.data.sum 
    }]); 

  //return group.getValue(0, 1); 
  return 10
}

function get_tweetsum2() 
{
  var group = google.visualization.data.group(graph_data, [{ 
    column: 0, 
    type: 'number', 
    // modify column 0 to return the same for all rows 
    // so we can get the sum of everything in column 1 
    modifier: function () { 
        return 0;
    }}], 
    [{ 
    // get the sum of column 1 
    column: 1, 
    type: 'number', 
    aggregation: google.visualization.data.sum 
    }]); 

  return group.getValue(0, 1); 
}

function update_frequency(data, highlight) {
  console.log('in highlight');
  second_last_idx = data['rows'].length-2;
  last_frequency = data['rows'][second_last_idx]['c'][1]['v'];
  $('span#frequency').html(last_frequency.toString());
  var frequency_text = 'in the past ' + interval_string[interval];
  $('span#frequency_text').html(frequency_text);
  if (highlight)
  {
    if (last_frequency > prev_last_frequency)
      $('span#frequency').addClass('positive');
    else if (last_frequency < prev_last_frequency)
      $('span#frequency').addClass('negative');
  
    setTimeout('unhighlight_frequency()', 2000);
  }
  prev_last_frequency = last_frequency;

  tweetsum = get_tweetsum2();
  daterange = graph_data.getColumnRange(0);

  $('span#last_date').html('as of '+ daterange.max.toLocaleTimeString());
  
  daterange = (daterange.max-daterange.min)/(interval*1000);
  average = (tweetsum/daterange).toFixed(0);
  $('span#average').html(average);

  freqrange = graph_data.getColumnRange(1);
  $('span#range').html('&nbsp;&nbsp; ' + freqrange.min.toString() + ' - ' + freqrange.max.toString());

}

function unhighlight_frequency() {
  console.log('in unhighlight');
  $('span#frequency').removeClass('positive');
  $('span#frequency').removeClass('negative');
}

function parse_graph(data) 
{
  for (i=0; i<data['rows'].length; i++)
  {
    utc_ts = 1000*data['rows'][i]['c'][0]['v'];
    utcdate = new Date(utc_ts);
    date = new Date(utc_ts - utcdate.getTimezoneOffset()*60*1000);
    data['rows'][i]['c'][0]['v'] = date;
  }
  console.log('Extra:', data['extra']);
  if (data['extra']['final'] == true)
    no_more_data = true;
  latest_tweet = data['extra']['latest_tweet']
  html = '<div style="padding: 2px 0px 2px 0px;">';
  html += '<div style="float: left;"><img border="0" width="40" height="40" src="';
  html += latest_tweet['profile_image_url'];
  html += '"/></div>';
  html += latest_tweet['tweet'];
  html += '<div style="clear: both;"></div></div>';
  $('#latest_tweets_div').html(html);
}

function toggle_graph()
{
  if (graph == graph1)
  {
    graph = graph2;
    console.log('Graph is now graph2');  
  }
  else if (graph == graph2)
  {
    graph = graph1;
    console.log('Graph is now graph1');  
  }
}
