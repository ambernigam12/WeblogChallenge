import numpy as np;

np.random.seed(12345)
import seaborn as sns;

sns.set(color_codes=True)
import pyspark
import shlex
import pandas as pd
from pandas import Series
from dateutil import parser
from datetime import timedelta

# Log File name
logFile = "output.log"

# configurable no. of instances to be displayed
numInstances = 2
sc = pyspark.SparkContext()

# Read weblog
rdd = sc.textFile("WeblogChallenge/data/2015_07_22_mktplace_shop_web_log_sample.log")

def appendLog(text):
    hs = open(logFile, "a")
    hs.write(text + "\n")
    hs.close()


# Parsing logs
def parse(x):
    keys = ['time', 'web_location', 'client_ip_port', 'backend_port', 'request_time', 'backend_time', 'response_time',
            'web_location_status', 'backend_status', 'received_bytes', 'sent_bytes', 'request', 'user_agent',
            'ssl_cipher', 'ssl_protocol']
    values = shlex.split(x)
    dictionary = dict({k: v for k, v in zip(keys, values)})
    return dictionary


rdd = rdd.map(lambda x: parse(x))

# Sessionize
def makeTimeWindows(mintime, maxtime, interval):
    timeWindows = []
    while mintime < maxtime:
        timeWindows.append((mintime, mintime + timedelta(minutes=interval)))
        mintime = mintime + timedelta(minutes=interval)
    return timeWindows


rdd.map(lambda x: x['time']).min(), rdd.map(lambda x: x['time']).max()
minTime = parser.parse(rdd.map(lambda x: x['time']).min())
maxTime = parser.parse(rdd.map(lambda x: x['time']).max())
timeWindows = makeTimeWindows(mintime=minTime, maxtime=maxTime, interval=3)


def getSessions(x, timeWindows=timeWindows):
    dt = parser.parse(x['time'])
    for i in range(len(timeWindows)):
        if dt >= timeWindows[i][0] and dt <= timeWindows[i][1]:
            x['session'] = str(i)
            break
    return x


session_data = rdd.map(lambda x: getSessions(x))
client_session = session_data.map(lambda x: ((x['client_ip_port'], x['session']), x))

# Average session time
average_time = (client_session
                .map(lambda x: (x[0], [x[1]['time']]))
                .reduceByKey(lambda x, y: x + y)
                .map(lambda x: (parser.parse(max(x[1])) - parser.parse(min(x[1]))).total_seconds())
                .mean())


# Unique URL hits per session
def processLists(x):
    return len(set(list(x)))


url = (client_session
       .map(lambda x: (x[0][1], x[1]['request']))
       .groupBy(lambda x: x[0])
       .map(lambda x: (x[0], processLists(x[1])))
       .collect())

# Most engaged user
top_users = (client_session
             .map(lambda x: (x[0], [x[1]['time']]))
             .reduceByKey(lambda x, y: x + y)
             .map(lambda x: (x[0], (parser.parse(max(x[1])) - parser.parse(min(x[1]))).total_seconds()))
             .takeOrdered(1, key=lambda x: -x[1]))


# Predict the expected load (requests/second) in the next minute
def rootMeanSquareError(predicted, actual):
    meanSquareError = (predicted - actual) ** 2
    return np.sqrt(meanSquareError.sum() / meanSquareError.count())


# Standard static time windows formation code
def makeTimeSecondsWindows(mintime, maxtime, interval):
    timeWindows = []
    while mintime < maxtime:
        timeWindows.append((mintime, mintime + timedelta(minutes=interval)))
        mintime = mintime + timedelta(seconds=interval)
    return timeWindows


minTime = parser.parse(rdd.map(lambda x: x['time']).min())
maxTime = parser.parse(rdd.map(lambda x: x['time']).max())
secondtimeWindows = makeTimeSecondsWindows(mintime=minTime, maxtime=maxTime, interval=1)
len(secondtimeWindows)


def matchSecondsTime(x, timeWindows=secondtimeWindows):
    dt = parser.parse(x)
    index = 0
    for i in range(len(timeWindows)):
        index = i
        if dt >= timeWindows[i][0] and dt < timeWindows[i][1]:
            break
    return index


requests_per_second = (client_session
                       .map(lambda x: x[1]['time'])
                       .map(lambda x: matchSecondsTime(x))
                       .map(lambda w: (w, 1))
                       .reduceByKey(lambda a, b: a + b)
                       .sortByKey())

ax = sns.tsplot(data=map(lambda x: x[1], requests_per_second.collect()))
dataFrame = pd.DataFrame()
dataFrame['crudeData'] = map(lambda x: x[0], requests_per_second.collect())
dataFrame['requestPerSecond'] = map(lambda x: x[1], requests_per_second.collect())
dataFrame['requestPerSecondLog'] = np.log(dataFrame['requestPerSecond'])
dataFrame['requestPerSecondMovingAverage'] = Series.rolling(dataFrame.requestPerSecondLog, window=12).mean()
dataFrame["requestPerSecondMovingAverageResults"] = np.exp(dataFrame.requestPerSecondMovingAverage)

#  Root Mean Squared Error
model_MovingAverage = rootMeanSquareError(dataFrame.requestPerSecondMovingAverageResults, dataFrame.requestPerSecond)

# Predict the session length for a given IP
ip_session_size = (client_session
                   .map(lambda x: (x[0], [x[1]['time']]))
                   .reduceByKey(lambda x, y: x + y)
                   .map(lambda x: (x[0], (parser.parse(max(x[1])) - parser.parse(min(x[1]))).total_seconds())))
ip_session_size_group_by_ip = (ip_session_size
                               .map(lambda x: (x[0][0], [x[1]]))
                               .reduceByKey(lambda x, y: x + y))


def trainMAModel(x):
    try:
        dataFrame = pd.DataFrame()
        dataFrame['sessionLength'] = x
        dataFrame['sessionLengthLog'] = np.log(dataFrame['sessionLength'])
        dataFrame['sessionLengthLogMovingAverage'] = Series.rolling(dataFrame.sessionLengthLog, window=1).mean()
        dataFrame["sessionLengthLogMovingAverage"] = np.exp(dataFrame.sessionLengthLogMovingAverage)
        model_rootMeanSquareError = rootMeanSquareError(dataFrame.sessionLengthLogMovingAverage,
                                                        dataFrame.sessionLength)
        return model_rootMeanSquareError
    except Exception as e:
        return 0.0


session_user = (ip_session_size_group_by_ip
                .map(lambda x: trainMAModel(x[1]))
                .filter(lambda x: x != 0)
                .mean())

# Predict the number of unique URL visits by a given IP
ip_url = (client_session
          .map(lambda x: (x[0], [x[1]['request']]))
          .reduceByKey(lambda x, y: x + y)
          .map(lambda x: (x[0], len(set(x[1]))))
          .map(lambda x: (x[0][0], [x[1]]))
          .reduceByKey(lambda x, y: x + y))
ip_unique_url_rootMeanSquareError = (ip_url
                                     .map(lambda x: trainMAModel(x[1]))
                                     .filter(lambda x: x != 0)
                                     .mean())

# Analytical results
appendLog("-----------------------Analytical Segment-----------------------")
appendLog("Average session time (in seconds): %s " % average_time)
appendLog("Unique URL visits per session: %s " % ip_url.take(numInstances))
appendLog("Most engaged user: %s " % top_users[0][0][0])
appendLog("-----------------------Analytical Segment-----------------------")

# Machine Learning tasks accuracy metrics
appendLog("-----------------------Machine Learning Segment-----------------------")
appendLog(
    "Accuracy metric (root mean square error) for predicting the expected load (requests/second) in the next minute: %s" % model_MovingAverage)
appendLog(
    "Accuracy metric (root mean square error) for predicting the session length for a given IP: %s" % session_user)
appendLog(
    "Accuracy metric (root mean square error) for predicting the number of unique URL visits by a given IP: %s" % ip_unique_url_rootMeanSquareError)
appendLog("-----------------------Machine Learning Segment-----------------------")

# Additional Info for verification
rdd.take(numInstances)
session_data.take(numInstances)
client_session.take(numInstances)
requests_per_second.take(numInstances)
ip_session_size.take(numInstances)
ip_session_size_group_by_ip.take(numInstances)
