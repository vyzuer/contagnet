#! /bin/csh

set imgsrc = '/home/vyzuer/work/data/DataSets/ftags/train/'
set src = '/hdfs/masl/tag_data/lmdb/train/'
set dst = '/data/masl/tag_data/lmdb/train/'
# set dst = '/home/vyzuer/work/data_local/ftags/data'
# set dst = '/media/SeSaMe_NAS/pandora_box_2/vyzuer/data/ftags/data/lmdb/'

python combine_label_con.py --imgsrc ${imgsrc} --src ${src} --dst ${dst}

