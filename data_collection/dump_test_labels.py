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

tag_db = None
yfcc100m = None
data_dir = None
num_splits = None
num_tags = None

sys.path.append('/home/vyzuer/work/code/ftags/')

import common.globals as gv


def load_globals():
    global tag_db
    global yfcc100m
    global data_dir
    global num_tags

    client = None

    host_name = socket.gethostname()
    if host_name == 'cassandra':
        client = MongoClient()
    else:
        client = MongoClient('172.29.35.126:27019')

    tag_db = client.flickr_tag
    yfcc100m = tag_db['yfcc100m']

    data_dir = gv.__dataset_path

    num_tags = gv.__NUM_TAGS_1540


def get_labels(pid):
    labels = np.zeros(num_tags, dtype='int')

    doc = yfcc100m.find_one({'_id': pid})
    tags = doc['User_tags']
    tag_col = tag_db['tags_1540']
    for t in tags.split(','):
        tag = tag_col.find_one({'_id': t})
        if tag is not None:
            _id = tag['label']
            labels[_id] = 1

    return labels

def master_label1540():

    col = tag_db['test_data']

    num_items = col.count()

    # store the labels in an array
    labels = np.zeros(shape=(num_items, num_tags), dtype='int')

    cursor = col.find(no_cursor_timeout=True)
    j = 0
    for doc in cursor:
        pid = doc['_id']

        # get the labels list
        labels[j,:] = get_labels(pid)

        j += 1

        
    cursor.close()

    base_dir = data_dir + '/DB/test/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    fname = base_dir + '/labels.mtx'
    if os.path.exists(fname):
        os.remove(fname)

    sparse_labels = sparse.csr_matrix(labels)
    io.mmwrite(fname, sparse_labels)


def master():

    col = tag_db['test_data']

    num_items = col.count()

    # store the labels in an array
    labels = np.zeros(shape=(num_items, num_tags), dtype='int')
    img_list = []

    cursor = col.find(no_cursor_timeout=True)
    j = 0
    for doc in cursor:
        pid = doc['_id']
        img_name = str(pid) + '.jpg'

        r_path = img_name[0:3] + '/' + img_name[3:6] + '/'
        img_path = r_path + img_name

        print img_path
        img_list.append(img_path)

        # get the labels list
        labels[j,:] = get_labels(pid)

        j += 1

        
    cursor.close()

    base_dir = data_dir + '/test/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    fname = base_dir + '/test.list'
    if os.path.exists(fname):
        os.remove(fname)
    np.savetxt(fname, img_list, fmt='%s')

    fname = base_dir + '/labels.mtx'
    if os.path.exists(fname):
        os.remove(fname)

    sparse_labels = sparse.csr_matrix(labels)
    io.mmwrite(fname, sparse_labels)
        

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print "Usage: python sys.argv[0]"
        exit(0)

    load_globals()

    master_label1540()

