#! /bin/csh

set src = '/home/vyzuer/work/data/DataSets/ftags/train'
set dst = '/hdfs/masl/tag_data/lmdb/train/'
# set dst = '/home/vyzuer/work/data_local/ftags/data'
# set dst = '/media/SeSaMe_NAS/pandora_box_2/vyzuer/data/ftags/data'

# foreach i ( `seq 0 5` )
foreach i (28)
    echo $i
    set img_src = "${src}/${i}/train_${i}.list"
    set img_dst = "${src}/${i}/train_${i}.list_new"
    set lables_src = "${src}/${i}/labels_1540_${i}.mtx"
    set lmdb_label = "${dst}/${i}/label_1540"
    
    python create_label.py \
            --images ${img_src} \
            --images_dst ${img_dst} \
            --labels ${lables_src} \
            --labelsOut ${lmdb_label} 

end
