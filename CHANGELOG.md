#### Change Log

- 10/25/17

  - finished training ssd_inception_v2 model with udacity real images
  - pushed the frozen_graph to github
  - added a juypter notebook to run the inference model with test images and shows the predicted classifcation with bounding boxes on the image
  - changed predictor.py to run through predictions with a series of images and output each prediction as a csv row
    - each prediction currently takes around 250ms on my macbook pro 2014 with a GeForce GT 750M GPU

- 10/24/17

  - alec
    - tested training using udacity-carnd-deep-learning ami on AWS p2.xlarge
    - added PYTHONPATH setup in README.md
    - encountered very slow training step performance using default AMI image
      - about 10 seconds per step
      - updated to tensorflow-gpu 1.3.0 and lowered training step timeot 1.5 sec
    - found a bug in supporting multiple classes and classes_text in generate_tfrecord.py
      - pushed the fix
      - updated the record files under training_data
    - ran a few trainining tests
      - changed ssd config to lower the maximum number of object detections from 100 to 50

- 10/22/17

  - alec

    - performed limited training on a macbook pro using ssd_inception_v2_coco_1106_2017 as a end-to-end test of the object detection pipeline from training to prediction

      - decreased batch_size from 24 to 12 in order to fit into the macbook pro's gpu

      - trained up to 410 steps and stopped.  Loss fluctuates around 2.0

      - will need to perform training on AWS with p2.xlarge, but got rejected by AWS as my AWS account is relatively new.

        ​