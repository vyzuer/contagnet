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
dump_dir = None
viz = False
classes = None

sys.path.append('/home/vyzuer/work/code/ftags/')

import common.globals as gv


def load_globals():
    global tag_db
    global data_dir
    global num_splits
    global num_tags
    global image_data
    global viz
    global dump_dir
    global classes

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
    num_tags = gv.__NUM_TAGS_1540
    dump_dir = gv.__dump_dir

    classes = np.loadtxt('/home/vyzuer/work/data/DataSets/ftags/tags_1540_list.txt', dtype=str)

def plot_hist(prec1, prec2, rec1, rec2, ids, method):

    n_tags = 20
    fname = dump_dir + 'top_tags_hist_' + str(method) + '.png'

    # baseline
    p1 = prec1[ids][:n_tags]
    p2 = prec2[ids][:n_tags]
    r1 = rec1[ids][:n_tags]
    r2 = rec2[ids][:n_tags]

    tags = classes[ids][:n_tags]
    print tags
    
    mpl.rcParams.update({'font.size': 22})
    plt.subplot(111, aspect=10)
    bar_width = 0.20
    index = np.arange(n_tags)
    plt.bar(index, p1, bar_width, color='r',label='p0')
    plt.bar(index+bar_width, r1, bar_width, color='g',label='r0')
    plt.bar(index+2*bar_width, p2, bar_width, color='b',label='p1')
    plt.bar(index+3*bar_width, r2, bar_width, color='c',label='r1')
    # plt.xlabel('User tag')
    plt.ylabel('Scores')
    lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=4, mode="expand", borderaxespad=0.)
    plt.xticks(index+2*bar_width, tags, rotation='vertical', fontsize=16)
    plt.tight_layout()
    plt.savefig(fname, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def master():
    global classes
    # load the per label precision recall values
    f1 = dump_dir + 'per_tag_0.list'
    f2 = dump_dir + 'per_tag_1.list'

    pr1 = np.loadtxt(f1)
    pr2 = np.loadtxt(f2)

    # load the dataset per tag information
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

    prec1 = pr1[:,0][tag_ids]
    rec1 = pr1[:,1][tag_ids]
    prec2 = pr2[:,0][tag_ids]
    rec2 = pr2[:,1][tag_ids]

    classes = classes[tag_ids]

    f1_1 = np.asarray([2*p1*r1/(p1 + r1) if p1 >0 or r1 > 0 else 0 for p1,r1 in zip(prec1, rec1)])
    f1_2 = np.asarray([2*p1*r1/(p1 + r1) if p1 >0 or r1 > 0 else 0 for p1,r1 in zip(prec2, rec2)])

    f1_gain = f1_2 - f1_1
    p_gain = prec2 - prec1
    r_gain = rec2 - rec1

    # sorted_ids = f1_1.argsort()[::-1]
    sorted_ids = prec1.argsort()[::-1]
    # sorted_ids = labels_f.argsort()[::-1]

    sorted_ids1 = f1_1.argsort()[::-1]
    sorted_ids2 = f1_2.argsort()[::-1]
    sorted_ids3 = f1_gain.argsort()[::-1]
    sorted_ids4 = p_gain.argsort()[::-1]
    sorted_ids5 = r_gain.argsort()[::-1]

    if viz:
        plot_hist(prec1, prec2, rec1, rec2, sorted_ids1, 0)
        plot_hist(prec1, prec2, rec1, rec2, sorted_ids2, 1)
        # plot_hist(prec1, prec2, rec1, rec2, sorted_ids3, 2)
        # plot_hist(prec1, prec2, rec1, rec2, sorted_ids4, 3)
        # plot_hist(prec1, prec2, rec1, rec2, sorted_ids5, 4)

    for ids in (sorted_ids3, sorted_ids4, sorted_ids5):
        tags = classes[ids][:50]
        print tags
        tags = classes[ids[::-1]][:50]
        print tags
    
    full_hist = labels_f[sorted_ids]/max(labels_f)

    prec1 /= prec2.sum()
    prec2 /= prec2.sum()
    rec1 /= rec2.sum()
    rec2 /= rec2.sum()
    f1_1 /= f1_2.sum()
    f1_2 /= f1_2.sum()

    prec1_ = np.cumsum(prec1[sorted_ids])
    prec2_ = np.cumsum(prec2[sorted_ids])
    rec1_ = np.cumsum(rec1[sorted_ids])
    rec2_ = np.cumsum(rec2[sorted_ids])
    f1_1_ = np.cumsum(f1_1[sorted_ids])
    f1_2_ = np.cumsum(f1_2[sorted_ids])

    if viz:
        mpl.rcParams.update({'font.size': 22})
        fname = dump_dir + 'prec_recall.png'
        p1, = plt.plot(prec1_, color='b', label='p-cf', linewidth=2.0)
        p2, = plt.plot(prec2_, color='y', label='p-ca', linewidth=2.0)
        r1, = plt.plot(rec1_, color='c', label='r-cf', linewidth=2.0)
        r2, = plt.plot(rec2_, color='m', label='r-ca', linewidth=2.0)
        f1, = plt.plot(f1_1_, color='r', label='f1-ca', linewidth=2.0)
        f2, = plt.plot(f1_2_, color='g', label='f1-ca', linewidth=2.0)
        plt.axis([0, 1540, 0, 1])
        plt.xlabel('user-tags')
        plt.ylabel('cumulative precision/recall')
        plt.legend([p1, p2, r1, r2, f1, f2], ['p-cf', 'p-ca', 'r-cf', 'r-ca', 'f1-cf', 'f1-ca'], loc='best')
        # plt.legend([p1, p2, r1, r2], ['p-cf', 'p-ca', 'r-cf', 'r-ca'], loc='best')
        # plt.legend([p1, p2], ['p-cf', 'p-ca'])
        # plt.show()
        plt.savefig(fname)

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print "Usage: python sys.argv[0]"
        exit(0)

    load_globals()

    master()

