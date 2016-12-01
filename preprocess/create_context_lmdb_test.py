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
batch_size = 10000

import common.globals as gv

def _init():
    global data_base

    data_base = gv.__image_test_data


def fillLmdb(context_file, context, images):
    cnt = 0

    context_db = lmdb.open(context_file, map_size=int(1e12))

    context_txn = context_db.begin(write=True)

    t0 = time.time()
    examples = zip(images, context)
    for in_idx, (image, context) in enumerate(examples):
        try:
            #saved image
            # img_src = data_base + image
            # im = Image.open(img_src)

            # im = np.array(im) # or load whatever ndarray you need
            # if len(im.shape) < 3:
            #     raise Exception('This is a B/W image.')

            # im = im[:,:,::-1]
            #save context
            context = np.array(context).astype(float).reshape(1,1,len(context))
            context_dat = caffe.io.array_to_datum(context)
            context_txn.put('{:0>10d}'.format(in_idx), context_dat.SerializeToString())
            cnt += 1
            
            # write batch
            if cnt%batch_size == 0:
                context_txn.commit()
                context_txn = context_db.begin(write=True)
                print 'saved batch: ', cnt

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            print e
            print image
            print "Skipped image and label with id {0}".format(in_idx)
        if in_idx%500 == 0:
            string_ = str(in_idx+1) + ' / ' + str(len(images))
            sys.stdout.write("%s\r" % string_)
            sys.stdout.flush()

    if cnt%batch_size != 0:
        context_txn.commit()
        print 'saved last batch: ', cnt

    context_db.close()

    print cnt

    print 'running time: ', time.time() - t0
    print "\nFilling lmdb completed"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create lmdb for caffe'
    )
    parser.add_argument(
        '--images',
        type=str,
        help='Images file',
        required=True
    )
    parser.add_argument(
        '--context',
        type=str,
        help='context file',
        required=True
    )
    parser.add_argument(
        '--contextOut',
        type=str,
        help='context lmdb',
        required=True
    )

    args = parser.parse_args()

    images = np.loadtxt(args.images, str)
    context = np.loadtxt(args.context)

    _init()

    print "Creating test set..."

    fillLmdb(
        context_file=args.contextOut,
        context=context,
        images=images[:])

    print "Completed."

