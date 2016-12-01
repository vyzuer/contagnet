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
    client = None

    host_name = socket.gethostname()
    if host_name == 'cassandra':
        client = MongoClient()
    else:
        client = MongoClient('172.29.35.126:27019')

    tag_db = client.flickr_tag
    yfcc100m = tag_db['yfcc100m']

    # set the default timeout for urllib
    socket.setdefaulttimeout(1)

def download_images(data, media_path):

    # load the dataset
    col = tag_db[data]

    cursor = col.find(no_cursor_timeout=True)

    for doc in cursor:
        pid = doc['_id']
        img_name = str(pid) + '.jpg'

        r_path = '/' + img_name[0:3] + '/' + img_name[3:6] + '/'
        img_base = media_path + r_path

        if not os.path.exists(img_base):
            os.makedirs(img_base)

        img_path = img_base + img_name

        print img_path

        if os.path.exists(img_path):
            continue

        entry = yfcc100m.find_one({'_id': pid})

        url = entry['Photo_download_URL']
        prefix, ext = os.path.splitext(url)
        url = prefix + '_m' + ext

        print url
         
        thread = myThread(url, img_path)
        thread.start()

    print 'waiting for threads to complete...'
    cursor.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: python sys.agv[0] location_name base_directory"

    data = str(sys.argv[1])
    base_dir = str(sys.argv[2])

    load_globals()

    media_path = base_dir + data
    if not os.path.exists(media_path):
        os.makedirs(media_path)

    download_images(data, media_path)

