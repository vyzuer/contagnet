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
import sys 
import os
import matplotlib.pyplot as plt

import numpy as np
import os.path as osp

from copy import copy

caffe_root = '/home/vyzuer/work/caffe/'
sys.path.append(caffe_root + 'python')
import caffe # If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

from caffe import layers as L, params as P # Shortcuts to define the net prototxt.

sys.path.append("pycaffe/layers") # the datalayers we will use are in this directory.
sys.path.append("pycaffe") # the tools file is in this folder

import tools #this contains some tools that we need

sys.path.append('/home/vyzuer/work/code/ftags/')

batch_size = 10
n_samples = 46700
clasess = None
photo_ids = None
dump_dir = None

import common.globals as gv

def _init():

    global classes
    global photo_ids
    global dump_dir

    dump_dir = gv.__dump_dir

    caffe.set_mode_gpu()
    caffe.set_device(0)

    classes = np.loadtxt('/home/vyzuer/work/data/DataSets/ftags/tags_1540_list.txt', dtype=str)
    photo_ids = np.loadtxt('/home/vyzuer/work/data/DataSets/ftags/test/test.list', dtype=str)


def evaluate_k(gt, est, k=5):
    """
        the est are the estimated labels
        """
    acc = 0.0
    prec = 0.0
    rec = 0.0
    tp = 0.0

    tag_ids = est.argsort()[::-1]

    for i in range(k):
        _id = tag_ids[i]
        if gt[_id] == 1:
            acc = 1.0
            tp += 1.0

    prec = tp/k
    rec = tp/np.sum(gt)

    print acc, prec, rec

    return acc, prec, rec
    

def check_baseline_accuracy(net, num_batches):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data[:,0,0,:]
        ests = np.zeros(gts.shape)
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)
    return acc / (1. * num_batches * batch_size)


def hamming_distance(gt, est):
    return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))

def check_accuracy(net):
    num_batches = n_samples/batch_size
    acc = 0.0
    prec = 0.0
    rec = 0.0
    acc1 = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data[:,0,0,:]
        ests = net.blobs['fc9_flickr_con'].data[:,:]
        print gts.shape, ests.shape
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
                acc += hamming_distance(gt, est > 0)
                a, p, r = evaluate_k(gt, est, 5)
                acc1 += a
                prec += p
                rec += r

    acc1 /= (1. * num_batches * batch_size)
    prec /= (1. * num_batches * batch_size)
    rec  /= (1. * num_batches * batch_size)
    print 'acc: %.6f, prec: %.6f, rec: %.6f' %(acc1, prec, rec)

    return acc / (1. * num_batches * batch_size)

def predict_tags(net):
    fname = dump_dir + 'prediction_vgg_5.tsv'
    fp = open(fname, 'w')
    num_batches = n_samples/batch_size
    for t in range(num_batches):
        net.forward()
        # gts = net.blobs['label'].data[0,0,:]
        for i in range(batch_size):
            # pred_label = net.blobs['fc8_flickr'].data[i, ...]
            pred_label = net.blobs['fc9_flickr_con'].data[i, ...]
            estlist = pred_label.argsort()[::-1][:5]
            pred = [classes[estlist]]
            ppath = photo_ids[t*batch_size + i]
            print ppath
            photo_id = os.path.splitext(os.path.split(ppath)[1])[0]
            fp.write('%s\t' %(photo_id))
            np.savetxt(fp, pred, fmt='%s', delimiter=',')

    fp.close()

def master(model_dir, viz = False):

    model_def = model_dir + 'test.prototxt'    
    # model_weights = caffe_root + 'models/finetune_flickr_tag/finetune_flickr_tag_1_2_iter_40000.caffemodel'
    model_weights = caffe_root + 'models/vgg_flickr_tag/vgg_flickr_tag_1_iter_340000.caffemodel'
    # model_weights = caffe_root + 'models/finetune_flickr_tag_2/finetune_flickr_tag_0_iter_525999.caffemodel'
    # model_weights = caffe_root + 'models/finetune_flickr_tag_2/finetune_flickr_tag_1_iter_240000.caffemodel'
    # model_weights = caffe_root + 'models/finetune_flickr_tag/finetune_flickr_tag_3_iter_80000.caffemodel'
    # model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    net = caffe.Net(str(model_def),      # defines the structure of the model
                    str(model_weights),  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    # net.step(1)
    # print 'Baseline accuracy:{0:.4f}'.format(check_baseline_accuracy(solver.test_nets[0], n_samples/batch_size))


    if viz:
        transformer = tools.SimpleTransformer() # This is simply to add back the bias, re-shuffle the color channels to RGB, and so on...
        image_index = 50 # First image in the batch.
        plt.figure()
        plt.imshow(transformer.deprocess(copy(net.blobs['data'].data[image_index, ...])))
        gtlist = net.blobs['label'].data[image_index, ...].astype(np.int)
        plt.title('GT: {}'.format(classes[np.where(gtlist[0,0,:])]))
        plt.axis('off');
        plt.show()

        for image_index in range(10,15):
            plt.figure()
            plt.imshow(transformer.deprocess(net.blobs['data'].data[image_index, ...]))
            gtlist = net.blobs['label'].data[image_index, ...].astype(np.int)[0,0,:]
            estlist = net.blobs['fc8_flickr'].data[image_index, ...].argsort()[::-1][:5]
            plt.title('GT: {} \n EST: {}'.format(classes[np.where(gtlist)], classes[estlist]))
            plt.axis('off')
            plt.show()

    predict_tags(net)

    # print 'accuracy:{0:.4f}'.format(check_accuracy(net))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print 'usage: python ', sys.argv[0], ' model_dir'
        exit(0)

    model_path = sys.argv[1]

    _init()

    master(model_path, viz=False)

