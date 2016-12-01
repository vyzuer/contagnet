#! /bin/csh

set src = '/home/vyzuer/work/data/DataSets/ftags/test/'
set dst = '/hdfs/masl/tag_data/lmdb/test/'
# set dst = '/home/vyzuer/work/data_local/ftags/data'
# set dst = '/media/SeSaMe_NAS/pandora_box_2/vyzuer/data/ftags/data'

set img_src = "${src}/test.list"
set lables_src = "${src}/labels.mtx"
set lmdb_data = "${dst}/data"
set lmdb_label = "${dst}/label"

python create_multilabel_lmdb_test.py \
        --images ${img_src} \
        --labels ${lables_src} \
        --imagesOut ${lmdb_data} \
        --labelsOut ${lmdb_label} \
        --maxPx 256 \
        --minPx 227 


