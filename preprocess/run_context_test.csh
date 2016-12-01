#! /bin/csh

set src = '/home/vyzuer/work/data/DataSets/ftags/test'
set dst = '/hdfs/masl/tag_data/lmdb/test/'
# set dst = '/home/vyzuer/work/data_local/ftags/data'
# set dst = '/media/SeSaMe_NAS/pandora_box_2/vyzuer/data/ftags/data'

# foreach i ( `seq 0 5` )
set img_src = "${src}/test.list"
set context_src = "${src}/context_2.list"
set lmdb_data = "${dst}/data_con_2"

python create_context_lmdb_test.py \
        --images ${img_src} \
        --context ${context_src} \
        --contextOut ${lmdb_data} \

