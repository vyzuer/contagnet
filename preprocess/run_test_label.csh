#! /bin/csh

set src = '/home/vyzuer/work/data/DataSets/ftags/test/'
set dst = '/hdfs/masl/tag_data/lmdb/test/'
# set dst = '/home/vyzuer/work/data_local/ftags/data'
# set dst = '/media/SeSaMe_NAS/pandora_box_2/vyzuer/data/ftags/data'

set img_src = "${src}/test.list"
set lables_src = "${src}/labels_1540.mtx"
set lmdb_label = "${dst}/label_1540"

python create_test_label.py \
        --images ${img_src} \
        --labels ${lables_src} \
        --labelsOut ${lmdb_label}


