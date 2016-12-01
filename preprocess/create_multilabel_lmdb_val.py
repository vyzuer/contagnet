from pymongo import MongoClient
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
import socket

batch_size = 10000
tag_db = None
yfcc2m = None
num_tags = None
num_dim = 6
data_base = None
lmdb_base = None


sys.path.append('/home/vyzuer/work/code/ftags/')

import common.globals as gv


def _init():
    global tag_db
    global yfcc2m
    global num_tags
    global data_base
    global lmdb_base

    data_base = gv.__image_data

    client = None

    host_name = socket.gethostname()
    if host_name == 'cassandra':
        client = MongoClient()
    else:
        client = MongoClient('172.29.35.126:27019')
        # client = MongoClient('localhost:27019')

    tag_db = client.flickr_tag
    yfcc2m = tag_db['yfcc2m_test']

    data_dir = gv.__dataset_path

    num_tags = gv.__NUM_TAGS
    lmdb_base = gv.__lmdb_val_base_dir


def resize(img, maxPx, minPx):
    try:
        width = img.size[0]
        height = img.size[1]
        smallest = min(width, height)
        largest = max(width, height)
        k = 1
        if largest > maxPx:
            k = maxPx / float(largest)
            smallest *= k
            largest *= k
        if smallest < minPx:
            k *= minPx / float(smallest)
        size = int(math.ceil(width * k)), int(math.ceil(height * k))
        img = img.resize(size, Image.ANTIALIAS)
        return img
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)


def get_labels(tags):
    labels = np.zeros(num_tags, dtype='int')

    tag_col = tag_db['tags_1540']
    for t in tags.split(','):
        doc = tag_col.find_one({'_id': t})
        _id = doc['label']

        labels[_id] = 1

    return labels


def fillLmdb(images_file, labels_file, context_file, userpref_file, maxPx, minPx):
    means = np.zeros(3)
    cnt = 0

    if not os.path.exists(lmdb_base):
        os.makedirs(lmdb_base)

    # clean the lmdb before creating one
    if images_file is not None:
        if os.path.exists(images_file):
            shutil.rmtree(images_file)
        os.makedirs(images_file)
    if labels_file is not None:
        if os.path.exists(labels_file):
            shutil.rmtree(labels_file)
        os.makedirs(labels_file)
    if context_file is not None:
        if os.path.exists(context_file):
            shutil.rmtree(context_file)
        os.makedirs(context_file)
    if userpref_file is not None:
        if os.path.exists(userpref_file):
            shutil.rmtree(userpref_file)
        os.makedirs(userpref_file)

    images_db = None
    labels_db = None
    context_db = None
    userpref_db = None

    if images_file is not None:
        images_db = lmdb.open(images_file, map_size=int(1e12))
    if labels_file is not None:
        labels_db = lmdb.open(labels_file, map_size=int(1e12))
    if context_file is not None:
        context_db = lmdb.open(context_file, map_size=int(1e12))
    if userpref_file is not None:
        userpref_db = lmdb.open(userpref_file, map_size=int(1e12))

    images_txn = None
    labels_txn = None
    context_txn = None
    userpref_txn = None

    if images_file is not None:
        images_txn = images_db.begin(write=True)
    if labels_file is not None:
        labels_txn = labels_db.begin(write=True)
    if context_file is not None:
        context_txn = context_db.begin(write=True)
    if userpref_file is not None:
        userpref_txn = userpref_db.begin(write=True)

    # documents to be deleted
    remove_list = []

    cursor = yfcc2m.find(no_cursor_timeout=True).sort('_id', 1)
    
    num_samples = yfcc2m.count()
    for in_idx, doc in enumerate(cursor):

        try:
            if images_file is not None:
                #save image
                pid = doc['_id']
                img_name = str(pid) + '.jpg'

                r_path = '/' + img_name[0:3] + '/' + img_name[3:6] + '/'
                img_src = data_base + r_path + img_name
                im = Image.open(img_src)

                img = resize(im, maxPx=maxPx, minPx=minPx)
                # img = im.resize((256,256), Image.ANTIALIAS)

                img = np.array(img) # or load whatever ndarray you need
                if len(img.shape) < 3:
                    raise Exception('This is a B/W image.')
                mean = img.mean(axis=0).mean(axis=0)
                means += mean
                img = img[:,:,::-1]
                img = img.transpose((2,0,1))
                im_dat = caffe.io.array_to_datum(img)
                images_txn.put('{:0>10d}'.format(cnt), im_dat.SerializeToString())

            if labels_file is not None:
                #save label
                label = get_labels(doc['tags'])
                label = np.array(label).astype(float).reshape(1,1,len(label))
                label_dat = caffe.io.array_to_datum(label)
                labels_txn.put('{:0>10d}'.format(cnt), label_dat.SerializeToString())
            
            if context_file is not None:
                #save context
                context = doc['norm_context']
                context = np.array(context).astype(float).reshape(1,1,len(context))
                context_dat = caffe.io.array_to_datum(context)
                context_txn.put('{:0>10d}'.format(cnt), context_dat.SerializeToString())

            if userpref_file is not None:
                userpref = doc['userpref']
                userpref = np.array(userpref).astype(float).reshape(1,1,len(userpref))
                userpref_dat = caffe.io.array_to_datum(userpref)
                userpref_txn.put('{:0>10d}'.format(cnt), userpref_dat.SerializeToString())

            cnt += 1

            im.close()

            # write batch
            if cnt%batch_size == 0:
                if images_file is not None:
                    images_txn.commit()
                    images_txn = images_db.begin(write=True)
                if labels_file is not None:
                    labels_txn.commit()
                    labels_txn = labels_db.begin(write=True)
                if context_file is not None:
                    context_txn.commit()
                    context_txn = context_db.begin(write=True)
                if userpref_file is not None:
                    userpref_txn.commit()
                    userpref_txn = userpref_db.begin(write=True)
                print 'saved batch: ', cnt

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            print e
            print "Skipped image and label with id {0}".format(in_idx)
            remove_list.append(pid)

        if in_idx%500 == 0:
            string_ = str(in_idx+1) + ' / ' + str(num_samples)
            sys.stdout.write("%s\r" % string_)
            sys.stdout.flush()

    if in_idx%batch_size != 0:
        if images_file is not None:
            images_txn.commit()
        if labels_file is not None:
            labels_txn.commit()
        if context_file is not None:
            context_txn.commit()
        if userpref_file is not None:
            userpref_txn.commit()
        print 'saved last batch: ', in_idx
  
    if images_file is not None:
        images_db.close()
    if labels_file is not None:
        labels_db.close()
    if context_file is not None:
        context_db.close()
    if userpref_file is not None:
        userpref_db.close()

    print "\nFilling lmdb completed"
    print "Image mean values for RGB: {0}".format(means / cnt)

    fmean = lmdb_base + '/rgb.mean'
    np.savetxt(fmean, means/cnt, fmt='%.4f')

    # delete the documents which are to be removed
    for pid in remove_list:
        yfcc2m.remove({'_id': pid})

    cursor.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create lmdb for caffe'
    )
    parser.add_argument(
        '--imagesOut',
        type=str,
        help='Images lmdb',
        required=False,
        default=None
    )
    parser.add_argument(
        '--labelsOut',
        type=str,
        help='Labels lmdb',
        default=None,
        required=False
    )
    parser.add_argument(
        '--contextOut',
        type=str,
        help='Context lmdb',
        default=None,
        required=False
    )
    parser.add_argument(
        '--userprefOut',
        type=str,
        help='Userpref lmdb',
        default=None,
        required=False
    )
    parser.add_argument(
        '-n',
        type=int,
        help='Number of test examples',
        default=500,
        required=False
    )
    parser.add_argument(
        '--maxPx',
        type=int,
        help='Max size of larger dimension after resize',
        required=True
    )
    parser.add_argument(
        '--minPx',
        type=int,
        help='Min size of smaller dimension after resize. Has higher priority than --max',
        required=True
    )

    args = parser.parse_args()

    _init()

    print "Creating validation set..."

    fillLmdb(
        images_file=args.imagesOut,
        labels_file=args.labelsOut,
        context_file=args.contextOut,
        userpref_file=args.userprefOut,
        minPx=args.minPx,
        maxPx=args.maxPx)

    print "Completed."

