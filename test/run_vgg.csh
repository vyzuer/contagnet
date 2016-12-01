#! /bin/csh

# set model = "/home/vyzuer/work/caffe/models/finetune_flickr_tag_2/"
set model = "/home/vyzuer/work/caffe/models/vgg_flickr_tag/"
# set model = "/home/vyzuer/work/caffe/models/flickr_tag/"

# python train.py ${model}
python test_vgg.py ${model}

