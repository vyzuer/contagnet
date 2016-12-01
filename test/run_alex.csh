#! /bin/csh

set model = "/home/vyzuer/work/caffe/models/alexnet_flickr_tag/"
# set model = "/home/vyzuer/work/caffe/models/flickr_tag/"

# python train.py ${model}
python test_alex.py ${model}

