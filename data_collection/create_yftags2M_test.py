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
from PIL import Image

tag_db = None
yfcc2m_test = None
yfcc100m = None
data_dir = None
num_splits = None
num_tags = None
num_dim = 6
max_items = 2000000
data_base = None


sys.path.append('/home/vyzuer/work/code/ftags/')

import common.globals as gv


def load_globals():
    global tag_db
    global yfcc100m
    global yfcc2m_test
    global data_dir
    global num_splits
    global num_tags
    global data_base

    data_base = gv.__image_data

    client = None

    host_name = socket.gethostname()
    if host_name == 'cassandra':
        client = MongoClient()
    else:
        client = MongoClient('172.29.35.126:27019')
        # client = MongoClient('localhost:27019')

    tag_db = client.flickr_tag
    yfcc100m = tag_db['yfcc100m']
    tag_db.drop_collection('yfcc2m_test')
    yfcc2m_test = tag_db['yfcc2m_test']

    data_dir = gv.__dataset_path

    num_splits = gv.__num_splits
    num_tags = gv.__NUM_TAGS


def _check_validity(pid, tags):

    flag = True
    doc = yfcc100m.find_one({'_id': pid})

    lat = doc['Latitude']
    lon = doc['Longitude']
    date = doc['Date_taken']
    if date is u'' or date == 'null':
        flag = False
        return flag, lat, lon, date

    if lat is u'':
        flag = False
        return flag, lat, lon, date
    if lon is u'':
        flag = False
        return flag, lat, lon, date

    tag_list = tags.split(',')
    if len(tag_list) < 5:
        flag = False
        return flag, lat, lon, date

    img_name = str(pid) + '.jpg'

    r_path = '/' + img_name[0:3] + '/' + img_name[3:6] + '/'
    image_path = r_path + img_name
    
    try:
        img_src = data_base + image_path
        img = Image.open(img_src)
        img = np.array(img) # or load whatever ndarray you need
        if len(img.shape) < 3:
            raise Exception('This is a B/W image.')
    except Exception as e:
        flag = False
    

    return flag, lat, lon, date

def get_collection(col_id):
    col_name = 'train_' + str(col_id)
    # load the dataset
    col = tag_db[col_name]

    return col



def master():

    j = 0
    for i in range(num_splits-2, num_splits):
        print 'Split id : %d\n'%(i)
        col = get_collection(i)
        k = 0
        num_items = col.count()

        cursor = col.find(no_cursor_timeout=True)
        for doc in cursor:
            pid = doc['_id']
            k += 1

            if j%100 == 0:
                sys.stdout.flush()
                stat1 = j*100./max_items
                stat2= k*100./num_items
                print 'status: [%.2f%%] [%.4f%%]\r'%(stat1,stat2),
                # print 'status: [%.2f%%] [%.2f%%]' %(stat1, stat2)

            # check image for validity
            flag, lat, lon, time = _check_validity(pid, doc['tags'])
            if flag == True:         
                j += 1
                yfcc2m_test.insert_one({"_id": doc['_id'], "uid": doc['uid'], "tags": doc['tags'], "Latitude": lat, "Longitude": lon, "Date_taken": time})

        sys.stdout.flush()
        stat1 = j*100./max_items
        stat2= k*100./num_items
        print 'status: [%.2f%%] [%.4f%%]\n'%(stat1,stat2)

        cursor.close()


if __name__ == "__main__":
    if len(sys.argv) != 1:
        print "Usage: python sys.argv[0]"
        exit(0)

    load_globals()

    master()

