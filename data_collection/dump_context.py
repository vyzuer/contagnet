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
yfcc100m = None
data_dir = None
num_splits = None
num_tags = None
num_dim = 6

sys.path.append('/home/vyzuer/work/code/ftags/')

import common.globals as gv


def load_globals():
    global tag_db
    global yfcc100m
    global data_dir
    global num_splits
    global num_tags

    client = None

    host_name = socket.gethostname()
    if host_name == 'cassandra':
        client = MongoClient()
    else:
        # client = MongoClient('172.29.35.126:27019')
        client = MongoClient('localhost:27019')

    tag_db = client.flickr_tag
    yfcc100m = tag_db['yfcc100m']

    data_dir = gv.__dataset_path

    num_splits = gv.__num_splits
    num_tags = gv.__NUM_TAGS


def get_collection(col_id):
    col_name = 'train_' + str(col_id)
    # load the dataset
    col = tag_db[col_name]

    return col


def get_context(pid):
    doc = yfcc100m.find_one({'_id': pid})

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

    # iterate for all the splits
    # for i in range(num_splits):
    for i in [4, 5, 6, 7, 8, 9]:
    # for i in [28]:
        col = get_collection(i)

        num_items = col.count()

        # store the labels in an array
        context = np.zeros(shape=(num_items, num_dim))

        cursor = col.find(no_cursor_timeout=True)
        j = 0
        for doc in cursor:
            pid = doc['_id']

            # get the labels list
            context[j,:] = get_context(pid)
            print j
            
            j += 1

        cursor.close()

        base_dir = data_dir + '/train/' + str(i)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # perform standard scaling
        scaler = StandardScaler().fit(context)
        context = scaler.transform(context)

        fname = base_dir + '/context_' + str(i) + '.list'
        np.savetxt(fname, context)

        spath = base_dir + '/scaler/'
        if not os.path.exists(spath):
            os.makedirs(spath)
        scaler_path = spath + '/scaler.pkl'
        joblib.dump(scaler, scaler_path)


if __name__ == "__main__":
    if len(sys.argv) != 1:
        print "Usage: python sys.argv[0]"
        exit(0)

    load_globals()

    master()

