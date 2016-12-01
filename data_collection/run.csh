#!/bin/csh

# shuffle and split the image list
# python split_shuffle_train_data.py

# dump image list and labels
# python dump_train_list.py

# dump test list and labels
# python dump_test_list.py

# base directory for storing all the media data
# set base_dir = /home/vyzuer/work/data/DataSets/tags/data/
set base_dir = /hdfs/masl/tag_data/

foreach data ("train_data")
# foreach data ("test_data")
    echo $data
    python download_images.py $data $base_dir
end

