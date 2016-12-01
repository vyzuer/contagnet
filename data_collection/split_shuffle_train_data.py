from pymongo import MongoClient
import numpy as np
import sys
import os
import urllib
import logging
import socket
import time
import random

tag_db = None
master_col = None
data_dir = None

sys.path.append('/home/vyzuer/work/code/ftags/')

import common.globals as gv

split_size = None
num_splits = 29

def load_globals():
    global tag_db
    global master_col
    global data_dir
    global split_size
    client = None

    host_name = socket.gethostname()
    if host_name == 'cassandra':
        client = MongoClient()
    else:
        client = MongoClient('172.29.35.126:27019')

    tag_db = client.flickr_tag
    master_col = tag_db['train_data']

    data_dir = gv.__dataset_path

    split_size = gv.__split_size


def get_collection(col_id):
    col_name = 'train_' + str(col_id)
    # load the dataset
    col = tag_db[col_name]

    # drop the collection
    col.drop()

    return col

def master():

    #if shuffle already done load list from disk

    img_list = None
    f_img_list = data_dir + '/train/tag-train-photo-shuffle.list'
    if not os.path.exists(f_img_list):
        # load the train images list
        print 'loading image list...'
        f_img_list = data_dir + '/train/tag-train-photo.list'
        img_list = np.loadtxt(f_img_list, dtype='int')
        print 'loading done.'

        # randomly shuffle the images and save the random shuffle for later use
        print 'shuffling image list...'
        random.shuffle(img_list)
        print 'shuffling done.'

        # dump this for later use
        print 'saving shuffled list...'
        f_img_list = data_dir + '/train/tag-train-photo-shuffle.list'
        np.savetxt(f_img_list, img_list, fmt='%d')
        print 'dump done.'
    else:
        print 'loading previously dumped image list...'
        f_img_list = data_dir + '/train/tag-train-photo-shuffle.list'
        img_list = np.loadtxt(f_img_list, dtype='int')
        print 'loading done.'
        
    # now split the dataset into 28 partitions and save in mongodb
    i = 1
    col_id = 0

    col = get_collection(col_id)
    print 'populating collection: ', col_id

    for img in img_list:
        doc = master_col.find_one({'_id': img})
        col.insert_one(doc)

        i += 1
        if i > split_size:
            col_id += 1
            i = 1
            col = get_collection(col_id)
            print 'populating collection: ', col_id

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print "Usage: python sys.argv[0]"
        exit(0)

    load_globals()

    master()

