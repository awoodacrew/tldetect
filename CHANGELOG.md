#### Change Log

- 10/22/17

  - alec

    - performed limited training on a macbook pro using ssd_inception_v2_coco_1106_2017 as a end-to-end test of the object detection pipeline from training to prediction

      - decreased batch_size from 24 to 12 in order to fit into the macbook pro's gpu

      - trained up to 410 steps and stopped.  Loss fluctuates around 2.0

      - will need to perform training on AWS with p2.xlarge, but got rejected by AWS as my AWS account is relatively new.

        ​