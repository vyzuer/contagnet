#! /bin/csh

set context = 1
set model = "/home/vyzuer/work/caffe/models/alexnet_flickr_tag/"

# set context = 0
# set model = "/home/vyzuer/work/caffe/models/finetune_flickr_tag/"

# python train.py ${model}
python evaluate.py ${model} ${context}

