#### Overview

This repo contains the minimum steps to run tensorflow object detection training and inferencing for traffic light detection in the final udacity system integration project. The repo incorporates annotated training data from udacity rosbag files and udacity project simulator shared by : 

[Anthony Sarkis]: https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI

#### Directory Structure

- tldetect/
  - deployment/, nets/, object_detection/
    - these direcotyr were copied from tensorflow/model commit:538f89c4121803645fe72f41ebfd7069c706d954
    - fixed was applied to object_detection/exporter.py (see commit log)
  - raw_annotated_data/ (from Anthony Sarkis)
  - generate_tfrecords.py
    - transform raw_annotated_data to TFExamples/TFRecords format for tensorflow training
  - models/
    - we can test different model by adding more directories here.  The first one is ssd_inception_v2_coco_11_06_2017_udacity_real
  - models/ssd_inception_v2_coco_11_06_2017_udacity_real
    - untar the tensorflow pre-trained model here
    - models/ssd_inception_v2_coco_11_06_2017_udacity_real/train
      - set the model training output directory to here in the model's config file
    - models/ssd_inception_v2_coco_11_06_2017_udacity_real/eval
      - set the model evaluation output directory to here in the model's config file
  - training_data/
    - contains outputs from generate_tfrecords.py
  - ssd_inception_v2_udacity_real_train.sh
    - a script to train a ssd_inception_v2 model using udacity real driving images
  - export_graph.sh
    - a script to export a trained model for inference
    - the inference model will be in a directory with the name defined in the script
  - frozen_graphs/
    - contains the trained model as a frozen graph for inference
    - each model & image type (real vs simulator) should be in its own directory
  - test_images_real/
    - contains a few real driving images to test the inference model
  - test_images_simulator/
    - contains a few driving images from the simulator to test the inference model
  - predictor.py
    - based on tensorflow's object_detection sample, this code takes an object detection inference model and output the prediction in csv format
  - Traffic Light Detection and Classification - Capstone Project.ipynb
    - a Juypter notebook to test inference models and see the resulting images with the detected bounding boxes and predictions
  - CHANGELOG.md
    - We share out progress here as we go.
- Set the PYTHONPATH


```shell
# run this at the path/tldetect directory
export PYTHONPATH=`pwd`:$PYTHONPATH 
```



#### Add new models to train and test

To train using different models

1. All models should use the same training data under training_data/ and test data under either test_images_real/ or test_images_simulator/ (to be added)

2. Under models/, create your own model directory for training and try to name it by adding the model_name and what kind of data: e.g. ssd_inception_v2_coco_11_06_2017_udacity_real

3. Create your own training script similar to ssd_inception_v2_udacity_real_train.sh.

4. Modify the object_detection pipeline configuration.  For ssd_inception_v2_udacity_real, the configuration file is under: 

   `models/ssd_inception_v2_coco_11_06_2017_udacity_real/ssd_inception_v2_coco.config`

5. Change or create your own export_graph.sh to generate the inference model directory. 

6. Change or improve predictor.py to test your model.





