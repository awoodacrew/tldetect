PATH_TO_YOUR_PIPELINE_CONFIG="./models/ssd_inception_v2_coco_11_06_2017_udacity_real/ssd_inception_v2_coco.config"
PATH_TO_TRAIN_DIR="./models/ssd_inception_v2_coco_11_06_2017_udacity_real/train"

python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --train_dir=${PATH_TO_TRAIN_DIR}
