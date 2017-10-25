import os
import collections
from glob import glob
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from PIL import Image


# What model to use
MODEL_NAME = 'ssd_inception_v2_coco_11_06_2017_udacity_real_inference_graph'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = "./frozen_graphs/" + MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training_data', 'tl_label_map.pbtxt')

NUM_CLASSES = 4

#PATH_TO_TEST_IMAGES_DIR = 'test_images_real'
PATH_TO_TEST_IMAGES_DIR = '../../our-team/Capstone/data/traffic_light/loop_with_traffic_light/'
TEST_IMAGE_PATHS = glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg'))

#TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'left0000.jpg'),
#                    os.path.join(PATH_TO_TEST_IMAGES_DIR, 'left0285.jpg'),
#                    os.path.join(PATH_TO_TEST_IMAGES_DIR, 'left0350.jpg')]

detection_graph = tf.Graph()

def load_frozen_model():
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')  

def load_label_map():
  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)  
  return categories, category_index

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def get_category(categories, index):
  for category in categories:
    if category['id'] == index:
      return category
  return None

def extract_box(box_tuple, image):
  ymin, xmin, ymax, xmax = box_tuple[0]
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, 
                                xmax * im_width, 
                                ymin * im_height, 
                                ymax * im_height) 
  return (int(left), int(right), int(top), int(bottom))

def predict(categories, category_index):
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      # Definite input and output Tensors for detection_graph
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
      detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      for image_path in TEST_IMAGE_PATHS:
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        #vis_util.visualize_boxes_and_labels_on_image_array(
        #    image_np,
        #    np.squeeze(boxes),
        #    np.squeeze(classes).astype(np.int32),
        #    np.squeeze(scores),
        #    category_index,
        #    use_normalized_coordinates=True,
        #    line_thickness=8)

        print(os.path.basename(image_path), end=",")

        top_score = scores[0][0]
        if top_score < 0.5:
          print("Unknown")
        else:
          class_index = int(classes[0][0])
          category = get_category(categories, class_index)
          if category is not None:
            print("{},{}".format(category['name'], top_score))

        #print("\tBounding box for the highest score object (l,r,t,b):{}".format(extract_box(tuple(boxes[0].tolist()), image)))
        #pick the highest score for now
        #for i in range(3):
        #  score = scores[0][i]
        #  class_index = int(classes[0][i])
        #  category = get_category(categories, class_index)
        #  if category is not None:
        #    print("\tRank:{} label:{} class_index:{} prob:{:0.4f}".format(i, category['name'], class_index, score))

        #counted = collections.Counter(classes.tolist()[0])
        #most_common_class_index=int(counted.most_common()[0][0])
        #category = get_category(categories, most_common_class_index)
        #if category is not None:
        #  print("\tMost commonly identified label:{}".format(category['name']))


def main(_):
  load_frozen_model()
  categories, category_index = load_label_map()
  predict(categories, category_index)

if __name__ == '__main__':
  tf.app.run()
