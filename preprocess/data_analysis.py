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
import matplotlib.pyplot as plt
import matplotlib as mpl

tag_db = None
data_dir = None
num_splits = None
num_tags = None
image_data = None
viz = False

sys.path.append('/home/vyzuer/work/code/ftags/')

import common.globals as gv


def load_globals():
    global tag_db
    global data_dir
    global num_splits
    global num_tags
    global image_data
    global viz

    client = None

    host_name = socket.gethostname()
    if host_name == 'cassandra':
        client = MongoClient()
        viz = True
    else:
        client = MongoClient('172.29.35.126:27019')

    tag_db = client.flickr_tag

    data_dir = gv.__dataset_path
    image_data = gv.__image_data

    num_splits = gv.__num_splits
    num_tags = gv.__NUM_TAGS


def master_label():
    # store image count for each label
    base_dir = data_dir + '/train/'
    fname = base_dir + 'label_freq.list'
    fname1 = base_dir + 'label_freq_0.list'

    assert os.path.exists(fname)
    assert os.path.exists(fname1)

    labels_freq = np.loadtxt(fname, delimiter='\t')
    labels_freq1 = np.loadtxt(fname1, delimiter='\t')

    tag_ids = np.nonzero(labels_freq)

    labels_f = labels_freq[tag_ids]
    labels_f1 = labels_freq1[tag_ids]

    num_tags = labels_f.size
    max_cnt = np.max(labels_f)
    min_cnt = np.min(labels_f)
    mean_cnt = np.mean(labels_f)

    max_cnt1 = np.max(labels_f1)
    min_cnt1 = np.min(labels_f1)
    mean_cnt1 = np.mean(labels_f1)

    f1 = base_dir + 'data_analysis.list'
    fp = open(f1, 'w')

    fp.write('number of tags: %d\nmaximum count: %d\nminimum count: %d\naverage: %.2f\n' %(num_tags, max_cnt, min_cnt, mean_cnt))
    fp.write('number of tags: %d\nmaximum count: %d\nminimum count: %d\naverage: %.2f\n' %(num_tags, max_cnt1, min_cnt1, mean_cnt1))

    top_tag_ids = labels_freq.argsort()[::-1]
    top_tag_ids1 = labels_freq1.argsort()[::-1]

    ftag = data_dir + 'tags_list.txt'
    tags_list = np.loadtxt(ftag, dtype='str')
    top_tags = tags_list[top_tag_ids[:10]]
    top_tags1 = tags_list[top_tag_ids1[:10]]

    fp.write('28M top tags\n')
    np.savetxt(fp, top_tags, fmt='%s')
    fp.write('1M top tags\n')
    np.savetxt(fp, top_tags1, fmt='%s')

    fp.close()

    sorted_ids = labels_f.argsort()[::-1]
    full_hist = labels_f[sorted_ids]
    zero_hist = labels_f1[sorted_ids]

    if viz:
        mpl.rcParams.update({'font.size': 22})
        fname = base_dir + 'tag_count.png'
        p1, = plt.plot(np.log(full_hist), color='r', label='~28M', linewidth=2.0)
        p2, = plt.plot(np.log(zero_hist), color='b', label='~1M', linewidth=2.0)
        plt.axis([0, 1540, 0, 15])
        plt.xlabel('user tags')
        plt.ylabel('ln(image_count)')
        plt.legend([p1, p2], ['~28M', '~1M'])
        # plt.show()
        plt.savefig(fname)

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print "Usage: python sys.argv[0]"
        exit(0)

    load_globals()

    master_label()

