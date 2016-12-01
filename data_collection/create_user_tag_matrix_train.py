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
user_tag_matrix = None
data_dir = None
num_splits = None
num_tags = None
num_dim = 6
max_users = 300000
tag_mapping = {}

sys.path.append('/home/vyzuer/work/code/ftags/')

import common.globals as gv


def _init_tag_list():
    global tag_mapping
    
    tag_col = tag_db['tags_1540']
    cursor = tag_col.find(no_cursor_timeout=True)

    for doc in cursor:
        tag_mapping[doc['_id']] = doc['label']
        
    cursor.close()


def load_globals():
    global tag_db
    global yfcc100m
    global data_dir
    global num_splits
    global num_tags
    global user_tag_matrix

    client = None

    host_name = socket.gethostname()
    if host_name == 'cassandra':
        client = MongoClient()
    else:
        client = MongoClient('172.29.35.126:27019')
        # client = MongoClient('localhost:27019')

    tag_db = client.flickr_tag
    yfcc100m = tag_db['yfcc100m']
    tag_db.drop_collection('usertag_matrix_train')
    user_tag_matrix = tag_db['usertag_matrix_train']

    data_dir = gv.__dataset_path

    num_splits = gv.__num_splits
    num_tags = gv.__NUM_TAGS

    _init_tag_list()


def get_collection(col_id):
    col_name = 'train_' + str(col_id)
    # load the dataset
    col = tag_db[col_name]

    return col

def get_labels(tags):

    labels = np.zeros(num_tags, dtype='int')

    for t in tags.split(','):
        _id = tag_mapping[t]

        labels[_id] = 1

    return labels

def master():

    labels = np.zeros(shape=(max_users, num_tags), dtype='int')
    # iterate for all the splits
    k = 0
    for i in range(0, num_splits-2):
        print 'split id: %d\n'%(i)
        col = get_collection(i)

        num_items = col.count()
        j = 0

        cursor = col.find(no_cursor_timeout=True)
        for doc in cursor:
            j += 1
            if j%100 == 0:
                sys.stdout.flush()
                stat = j*100./num_items
                print 'status: [%.2f%%]\r'%(stat),

            uid = doc['uid']
            tags = doc['tags']
            user_id = k

            user = user_tag_matrix.find_one({'_id': uid})
            
            if user is None:
                user_tag_matrix.insert_one({'_id': uid, 'uid': k})
                k += 1
            else:
                user_id = user['uid']

            labels[user_id,:] += get_labels(tags)


        cursor.close()

        sys.stdout.flush()
        stat = j*100./num_items
        print 'status: [%.2f%%]\n'%(stat)


    # iterate through all the users and update database for labels
    cursor = user_tag_matrix.find(no_cursor_timeout=True)

    for doc in cursor:

        uid = doc['uid']
        user_tag_matrix.update_one({'_id': doc['_id']}, {'$set':{'labels': labels[uid,:].tolist()}})

    cursor.close()


if __name__ == "__main__":
    if len(sys.argv) != 1:
        print "Usage: python sys.argv[0]"
        exit(0)

    load_globals()

    master()

