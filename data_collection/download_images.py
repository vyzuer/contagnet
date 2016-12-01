from pymongo import MongoClient
import numpy as np
import sys
import os
import urllib
import logging
import socket
import time
import multiprocessing

tag_db = None
yfcc100m = None
yfcc100m_hash = None
s3_base = None

num_splits = None
split_size = None

sys.path.append('/home/vyzuer/work/code/ftags/')

import common.globals as gv

class myThread(multiprocessing.Process):
    def __init__(self, url, img_path):
        multiprocessing.Process.__init__(self)
        self.url = url
        self.img_path = img_path
    def run(self):
        download_image(self.url, self.img_path)

def download_image(url, img_path):
    # download image
    try:
        urllib.urlretrieve(url, filename=img_path)
    except Exception, e:
        logging.warn("error downloading %s: %s" % (url, e))


def load_globals():
    global tag_db
    global yfcc100m
    global yfcc100m_hash
    global s3_base
    global num_splits
    global split_size

    client = None

    host_name = socket.gethostname()
    if host_name == 'cassandra':
        client = MongoClient()
    else:
        client = MongoClient('172.29.35.126:27019')

    tag_db = client.flickr_tag
    yfcc100m = tag_db['yfcc100m']
    yfcc100m_hash = tag_db['yfcc100m_hash']
    s3_base = 'https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images/'

    # set the default timeout for urllib
    socket.setdefaulttimeout(2)

    num_splits = gv.__num_splits
    split_size = gv.__split_size

def download_images(data, media_path):

    # load the dataset
    col = tag_db[data]

    cursor = col.find(no_cursor_timeout=True)
    n_items = cursor.count()

    i = 0
    url = None
    for doc in cursor:
        pid = doc['_id']
        img_name = str(pid) + '.jpg'

        r_path = '/' + img_name[0:3] + '/' + img_name[3:6] + '/'
        img_base = media_path + r_path

        if not os.path.exists(img_base):
            os.makedirs(img_base)

        img_path = img_base + img_name

        # print img_path

        if i%1000 == 0:
            print ('status: %s %.4f%%\n' %(data, i*100./n_items))

        if os.path.exists(img_path):
            if os.stat(img_path).st_size > 10000:
                i += 1
                continue
            else:
                # get the image from amazon repositery
                hid_doc = yfcc100m_hash.find_one({'_id': pid})
                hid = hid_doc['hash_id']
                url = s3_base + hid[0:3] + '/' + hid[3:6] + '/' + str(hid) + '.jpg'
                # print url
        else:
            entry = yfcc100m.find_one({'_id': pid})

            url = entry['Photo_download_URL']
            prefix, ext = os.path.splitext(url)
            url = prefix + '_m' + ext

        # print url
         
        thread = myThread(url, img_path)
        thread.start()

        i += 1

    print 'waiting for threads to complete...'
    cursor.close()

def master_test(media_path, data='test_data'):

    download_images(data, media_path)


def master_train(media_path):

    # for i in range(28, num_splits):
    for i in range(10, 28):
        col_name = 'train_' + str(i)
        print col_name
        download_images(col_name, media_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: python sys.agv[0] data base_directory"
        exit(0)

    data = str(sys.argv[1])
    base_dir = str(sys.argv[2])

    load_globals()

    media_path = base_dir + data
    if not os.path.exists(media_path):
        os.makedirs(media_path)

    if data == 'train_data':
        print 'downloading train data...'
        master_train(media_path)
    else:
        print 'downloading test data...'
        master_test(media_path, data)


