from pymongo import MongoClient
import numpy as np
import sys
import os
import urllib
import logging
import socket
import time
import random
from scipy import sparse, io
from dateutil import parser
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

tag_db = None
yfcc2m = None
data_dir = None
num_tags = None
num_dim = 6
num_samples = 2000000

sys.path.append('/home/vyzuer/work/code/ftags/')

import common.globals as gv


def load_globals():
    global tag_db
    global data_dir
    global num_splits
    global num_tags
    global yfcc2m

    client = None

    host_name = socket.gethostname()
    if host_name == 'cassandra':
        client = MongoClient()
    else:
        # client = MongoClient('172.29.35.126:27019')
        client = MongoClient('localhost:27019')

    tag_db = client.flickr_tag
    yfcc2m = tag_db['yfcc2m_test']

    data_dir = gv.__database_path

    num_tags = gv.__NUM_TAGS


def get_context(doc):

    lat = doc['Latitude']
    lon = doc['Longitude']
    date = doc['Date_taken']
    if date is u'' or date == 'null':
        date = datetime.today()
    else:
        date = parser.parse(date)

    if lat is u'':
        lat = 0.0
    if lon is u'':
        lon = 0.0
    
    context = [lat, lon, date.minute + date.hour*60, date.day, date.month, date.weekday()]
    return context

def master():


    cursor = yfcc2m.find(no_cursor_timeout=True).sort('_id', 1)

    # store the labels in an array
    num_samples = cursor.count()
    context = np.zeros(shape=(num_samples, num_dim))
    for idx, doc in enumerate(cursor):
        # get the context
        context[idx,:] = get_context(doc)
        
    cursor.close()

    base_dir = data_dir + '/train/' 
    spath = base_dir + '/scaler/'
    scaler_path = spath + '/scaler.pkl'
    scaler = joblib.load(scaler_path)

    # perform standard scaling
    context = scaler.transform(context)

    cursor = yfcc2m.find(no_cursor_timeout=True).sort('_id', 1)
    for idx, doc in enumerate(cursor):
        # update the normalized context for later use
        yfcc2m.update_one({'_id': doc['_id']}, {'$set':{'norm_context': context[idx,:].tolist()}})
        
        
    cursor.close()


if __name__ == "__main__":
    if len(sys.argv) != 1:
        print "Usage: python sys.argv[0]"
        exit(0)

    load_globals()

    master()

