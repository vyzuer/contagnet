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

batch_size = 100
n_samples = 154800
# n_samples = 10000
clasess = None
photo_ids = None
dump_dir = None
num_asamples = 0

# these variables are for measuring individual tag accuracy
tp_ptag = None # true positive rate
ap_ptag = None # prediction rate
total_ptag = None # total present

n_tags = None

import common.globals as gv

def _init(context):

    global classes
    global photo_ids
    global dump_dir
    global n_tags

    caffe.set_mode_gpu()
    caffe.set_device(0)

    # photo_ids = np.loadtxt('/home/vyzuer/work/data/DataSets/ftags/test/test.list', dtype=str)
    photo_ids = np.loadtxt('/home/vyzuer/work/data/DataSets/ftags/train/28/train_28.list_new', dtype=str)

    dump_dir = gv.__dump_dir

    if context:
        n_tags = gv.__NUM_TAGS_1540
        classes = np.loadtxt('/home/vyzuer/work/data/DataSets/ftags/tags_1540_list.txt', dtype=str)
    else:
        n_tags = gv.__NUM_TAGS
        classes = np.loadtxt('/home/vyzuer/work/data/DataSets/ftags/tags_list.txt', dtype=str)

def evaluate_k(gt, est, k=5):
    """
        the est are the estimated labels
        """
    global tp_ptag 
    global ap_ptag 
    global total_ptag

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

    if k == 5:
        num_tags = np.sum(gt)
        for i in range(num_tags):
            _id = tag_ids[i]
            ap_ptag[_id] += 1
            if gt[_id] == 1:
                tp_ptag[_id] += 1
            
        total_ptag += gt

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

def check_accuracy_nonc(net, k=5):
    global num_asamples
    num_asamples = 0
    num_batches = n_samples/batch_size
    acc = 0.0
    prec = 0.0
    rec = 0.0
    acc1 = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data[:,0,0,:]
        ests = net.blobs['fc8_flickr'].data[:,:]
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            if np.sum(gt) > 4:
                num_asamples += 1
                acc += hamming_distance(gt, est > 0)
                a, p, r = evaluate_k(gt, est, k)
                acc1 += a
                prec += p
                rec += r

    acc1 /= (1. * num_asamples)
    prec /= (1. * num_asamples)
    rec  /= (1. * num_asamples)
    print 'acc: %.6f, prec: %.6f, rec: %.6f' %(acc1, prec, rec)

    return np.array([acc1, prec, rec])

def check_accuracy(net, k=5, context=1):
    global num_asamples
    pred_layer = None
    if context:
        pred_layer = 'fc9_flickr_con'
    else:
        pred_layer = 'fc8_flickr'

    fname = dump_dir + 'prediction_val_' + str(k) + '_' + str(context) + '.tsv'
    fp = open(fname, 'w')
    num_asamples = 0
    num_batches = n_samples/batch_size
    acc = 0.0
    prec = 0.0
    rec = 0.0
    acc1 = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data[:,0,0,:]
        ests = net.blobs[pred_layer].data[:,:]
        i = 0
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            if np.sum(gt) > 4:
                num_asamples += 1
                a, p, r = evaluate_k(gt, est, k)
                acc1 += a
                prec += p
                rec += r

                estlist = est.argsort()[::-1][:5]
                pred = [classes[estlist]]
                ppath = photo_ids[t*batch_size + i]
                # print ppath
                photo_id = os.path.splitext(os.path.split(ppath)[1])[0]
                fp.write('%s\t%.3f\t%.3f\t%.3f\t' %(photo_id, a, p, r))
                np.savetxt(fp, pred, fmt='%s', delimiter=',')

            i += 1

    acc1 /= (1. * num_asamples)
    prec /= (1. * num_asamples)
    rec  /= (1. * num_asamples)
    print 'acc: %.6f, prec: %.6f, rec: %.6f' %(acc1, prec, rec)
    print 'total samples: ', num_asamples

    fp.close()

    return np.array([acc1, prec, rec])

def predict_tags(net):
    fp = open('prediction.tsv', 'w')
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
            # print ppath
            photo_id = os.path.splitext(os.path.split(ppath)[1])[0]
            fp.write('%s\t' %(photo_id))
            np.savetxt(fp, pred, fmt='%s', delimiter=',')

    fp.close()

def master_nonc(model_dir, viz = False):

    model_def = model_dir + 'validate.prototxt'    
    model_weights = caffe_root + 'models/finetune_flickr_tag/finetune_flickr_tag_1_iter_680000.caffemodel'
    # model_weights = caffe_root + 'models/finetune_flickr_tag_2/finetune_flickr_tag_0_iter_357194.caffemodel'
    # model_weights = caffe_root + 'models/finetune_flickr_tag/finetune_flickr_tag_3_iter_80000.caffemodel'
    # model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
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

    # predict_tags(net)

    fp = open('results_nonc.list', 'w')
    for k in (1, 3, 5, 10):
        res = check_accuracy_nonc(net, k)
        fp.write('k: %d\n' %(k))
        fp.write('accuracy precision recall\n')
        np.savetxt(fp, res, fmt='%.6f')
    fp.close()

def master(model_dir, context, viz = False):

    global tp_ptag 
    global ap_ptag 
    global total_ptag

    tp_ptag = np.zeros(n_tags)
    ap_ptag = np.zeros(n_tags)
    total_ptag = np.zeros(n_tags)

    model_def = model_dir + 'validate.prototxt'    
    model_weights = None
    # model_weights = caffe_root + 'models/finetune_flickr_tag/finetune_flickr_tag_1_iter_120000.caffemodel'
    if context:
        model_weights = caffe_root + 'models/alexnet_flickr_tag/alexnet_flickr_tag_3_iter_180000.caffemodel'
    else:
        model_weights = caffe_root + 'models/finetune_flickr_tag/finetune_flickr_tag_1_iter_680000.caffemodel'
    # model_weights = caffe_root + 'models/finetune_flickr_tag/finetune_flickr_tag_3_iter_80000.caffemodel'
    # model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    # net.step(1)
    # print 'Baseline accuracy:{0:.4f}'.format(check_baseline_accuracy(solver.test_nets[0], n_samples/batch_size))

    # predict_tags(net)

    fname = dump_dir + 'results_' + str(context) + '.list'
    fp = open(fname, 'w')
    # for k in (1, 3, 5, 10):
    for k in ([5]):
        res = check_accuracy(net, k, context)

        fp.write('k: %d\n' %(k))
        fp.write('accuracy precision recall\n')
        np.savetxt(fp, res, fmt='%.6f')

    fp.close()

    # dump the per tag accuracy results
    prec = [x/y if y else 0 for x,y in zip(tp_ptag,ap_ptag)]
    rec = [x/y if y else 0 for x,y in zip(tp_ptag,total_ptag)]

    pt_acc = np.vstack([prec, rec]).transpose()
    fname_pt = dump_dir + 'per_tag_' + str(context) + '.list'
    np.savetxt(fname_pt, pt_acc, fmt='%.6f')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print 'usage: python ', sys.argv[0], ' model_dir', 'context_mode'
        exit(0)

    model_path = sys.argv[1]
    context = int(sys.argv[2])

    _init(context)

    master(model_path, context=context, viz=False)
    # master_nonc(model_path, viz=False)

