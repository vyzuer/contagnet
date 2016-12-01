#! /bin/csh

set src = '/home/vyzuer/work/data/DataSets/ftags/train'
set dst = '/hdfs/masl/tag_data/lmdb/train/'
# set dst = '/home/vyzuer/work/data_local/ftags/data'
# set dst = '/media/SeSaMe_NAS/pandora_box_2/vyzuer/data/ftags/data'

# foreach i ( `seq 0 5` )
foreach i ( 0 1 2 3 )
    echo $i
    set img_src = "${src}/${i}/train_${i}.list"
    set context_src = "${src}/${i}/context_${i}.list"
    set lmdb_data = "${dst}/${i}/data_con"
    
    python create_context_lmdb.py \
            --images ${img_src} \
            --context ${context_src} \
            --contextOut ${lmdb_data} \

end
