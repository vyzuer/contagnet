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
import shutil

tag_db = None
data_dir = None
num_splits = None
num_tags = None
image_data = None

sys.path.append('/home/vyzuer/work/code/ftags/')

import common.globals as gv


def load_globals():
    global tag_db
    global data_dir
    global num_splits
    global num_tags
    global image_data

    client = None

    host_name = socket.gethostname()
    if host_name == 'cassandra':
        client = MongoClient()
    else:
        client = MongoClient('172.29.35.126:27019')

    tag_db = client.flickr_tag

    data_dir = gv.__dataset_path
    image_data = gv.__image_data

    num_splits = gv.__num_splits
    num_tags = gv.__NUM_TAGS_1540


def get_collection(col_id):
    col_name = 'train_' + str(col_id)
    # load the dataset
    col = tag_db[col_name]

    return col


def get_labels(tags):
    labels = np.zeros(num_tags, dtype='int')

    tag_col = tag_db['tags_1540']
    for t in tags.split(','):
        doc = tag_col.find_one({'_id': t})
        _id = doc['label']

        labels[_id] = 1

    return labels

def master_label():
    # store image count for each label
    labels_freq = np.zeros(num_tags, dtype='int')

    # iterate for all the splits
    # for i in range(num_splits):
    for i in range(1):
        col = get_collection(i)

        num_items = col.count()

        cursor = col.find(no_cursor_timeout=True)
        j = 0
        for doc in cursor:
            pid = doc['_id']

            # get the labels list
            labels = get_labels(doc['tags'])
            labels_freq += labels

            j += 1
	    print i, j
            
        cursor.close()


    base_dir = data_dir + '/train/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    fname = base_dir + 'label_freq_0.list'
    np.savetxt(fname, labels_freq, fmt='%d')

def master_label1540():

    # using the 28 collection for validation
    col = get_collection(28)

    num_items = col.count()

    # store the labels in an array
    labels = np.zeros(shape=(num_items, num_tags), dtype='int')

    j = 0

    cursor = col.find(no_cursor_timeout=True)
    for doc in cursor:
        pid = doc['_id']

        # get the labels list
        labels[j,:] = get_labels(doc['tags'])

        j += 1
        
    cursor.close()

    base_dir = data_dir + '/DB/val/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    fname = base_dir + '/labels.mtx'
    if os.path.exists(fname):
        os.remove(fname)

    sparse_labels = sparse.csr_matrix(labels)
    io.mmwrite(fname, sparse_labels)


def master():

    # iterate for all the splits
    for i in range(num_splits):
        col = get_collection(i)

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
            labels[j,:] = get_labels(doc['tags'])

            j += 1

            
        cursor.close()

        base_dir = data_dir + '/train/' + str(i)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        fname = base_dir + '/train_' + str(i) + '.list'
        if os.path.exists(fname):
            os.remove(fname)
        np.savetxt(fname, img_list, fmt='%s')

        fname = base_dir + '/labels_' + str(i) + '.mtx'
        if os.path.exists(fname):
            os.remove(fname)

        sparse_labels = sparse.csr_matrix(labels)
        io.mmwrite(fname, sparse_labels)
        
def master_copy_val():

    i = 28
    col = get_collection(i)

    base_dir = data_dir + '/data/train/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    num_items = col.count()

    cursor = col.find(no_cursor_timeout=True)
    for doc in cursor:
        pid = doc['_id']

        img_name = str(pid) + '.jpg'
        r_path = img_name[0:3] + '/' + img_name[3:6] + '/'

        img_path = r_path + img_name

        img_src = image_data + img_path

        dst_base = base_dir + r_path
        if not os.path.exists(dst_base):
            os.makedirs(dst_base)
        img_dst = dst_base + img_name

        print img_path

        shutil.copyfile(img_src, img_dst)

        
    cursor.close()


if __name__ == "__main__":
    if len(sys.argv) != 1:
        print "Usage: python sys.argv[0]"
        exit(0)

    load_globals()

    master_label1540()


