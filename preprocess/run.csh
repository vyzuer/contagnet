#! /bin/csh

# source of images for pixel values
set src = '/hdfs/masl/tag_data/train_data/'
# destination for dumping the lmdb data
set dst = '/data/masl/tag_data/lmdb/train/'

set lmdb_idata = "${dst}/idata"
set lmdb_label = "${dst}/label"
set lmdb_context = "${dst}/context"
set lmdb_userpref = "${dst}/userpref"

python create_multilabel_lmdb.py \
#         --imagesOut ${lmdb_idata} \
#         --labelsOut ${lmdb_label} \
        --contextOut ${lmdb_context} \
# --userprefOut ${lmdb_userpref} \
        --maxPx 256 \
        --minPx 227 


