from PIL import Image
import numpy as np
import lmdb
import caffe
import sys
import math
import argparse
import random
from scipy import sparse, io
import os
import time
import shutil

sys.path.append('/home/vyzuer/work/code/ftags/')

data_base = None
batch_size = 50000

import common.globals as gv

def _init():
    global data_base

    data_base = gv.__image_data


def combine_lmdb(imgsrc, src, dst):

    base_dir = dst + "combined/"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)

    # new database
    new_data_file = base_dir + "data"
    new_label_file = base_dir + "label_1540"
    new_con_file = base_dir + "data_con"
    new_image_list = base_dir + "image.list"

    new_data_db = lmdb.open(new_data_file, map_size=int(1e12))
    new_label_db = lmdb.open(new_label_file, map_size=int(1e12))
    new_con_db = lmdb.open(new_con_file, map_size=int(1e12))
    fp = open(new_image_list, 'w')

    data_txn = new_data_db.begin(write=True)
    label_txn = new_label_db.begin(write=True)
    con_txn = new_con_db.begin(write=True)

    i = 0
    j = 0
    l = 0
    for split in (2, 3):
        print "starting ", split

        src_d_file = src + str(split) + "/data"
        src_d_db = lmdb.open(src_d_file, readonly=True)
        src_d_txn = src_d_db.begin().cursor()
        
        print "starting data..."
        for (k, value) in src_d_txn:
            key = '{:0>10d}'.format(l)
            data_txn.put(key, value)
            l += 1

            # write batch
            if l%batch_size == 0:
                data_txn.commit()
                data_txn = new_data_db.begin(write=True)
                print 'saved batch: ', l


        data_txn.commit()
        data_txn = new_data_db.begin(write=True)
        src_d_db.close()

        src_l_file = src + str(split) + "/label_1540"
        src_l_db = lmdb.open(src_l_file, readonly=True)
        src_l_txn = src_l_db.begin().cursor()
        
        print "starting labels..."
        for (k, value) in src_l_txn:
            key = '{:0>10d}'.format(i)
            label_txn.put(key, value)
            i += 1

        label_txn.commit()
        label_txn = new_label_db.begin(write=True)
        src_l_db.close()

        src_c_file = src + str(split) + "/data_con"
        src_c_db = lmdb.open(src_c_file, readonly=True)
        src_c_txn = src_c_db.begin().cursor()
        
        print "starting context..."
        for (k, value) in src_c_txn:
            key = '{:0>10d}'.format(j)
            con_txn.put(key, value)
            j += 1

        con_txn.commit()
        con_txn = new_con_db.begin(write=True)
        src_c_db.close()

        print "starting images..."
        # write the image list for image data
        img_src = imgsrc + str(split) + "/train_" + str(split) + ".list_new"
        img_list = np.loadtxt(img_src, dtype=str)
        for img in img_list:
            img_path = data_base + img
            fp.write('%s %d\n' %(img_path, 0))

    print i, j
    new_data_db.close()
    new_label_db.close()
    new_con_db.close()
    fp.close()

    print "\nFilling lmdb completed"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cobine lmdb for caffe'
    )
    parser.add_argument(
        '--imgsrc',
        type=str,
        help='image src location',
        required=True
    )
    parser.add_argument(
        '--src',
        type=str,
        help='src location',
        required=True
    )
    parser.add_argument(
        '--dst',
        type=str,
        help='dst location',
        required=True
    )

    args = parser.parse_args()

    print "combining lmdb..."

    _init()

    combine_lmdb(imgsrc=args.imgsrc, src=args.src, dst=args.dst)

    print "Completed."

