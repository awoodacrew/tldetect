PATH_TO_PIPELINE_CONFIG="./models/ssd_inception_v2_coco_11_06_2017_udacity_real/ssd_inception_v2_coco.config"
PATH_TO_TRAIN_DIR="./models/ssd_inception_v2_coco_11_06_2017_udacity_real/train/model.ckpt-410"
PATH_TO_EXPORT_DIRECTORY="./frozen_graphs/ssd_inception_v2_coco_11_06_2017_udacity_real_inference_graph"

python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PATH_TO_PIPELINE_CONFIG} \
    --trained_checkpoint_prefix ${PATH_TO_TRAIN_DIR} \
    --output_directory ${PATH_TO_EXPORT_DIRECTORY}

