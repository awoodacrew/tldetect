import yaml
import os
import io
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util

REAL_TRAINING_DATA_DIR = 'raw_annotated_data/real_training_data'
REAL_TRAINING_DATA_YAML = 'real_data_annotations.yaml'
SIM_TRAINING_DATA_DIR = 'raw_annotated_data/sim_training_data'
SIM_TRAINING_DATA_YAML = 'sim_data_annotations.yaml'

def class_text_to_int(label):
  if label == 'Green':
    return 3
  if label == 'Yellow':
    return 2
  if label == 'Red':
    return 1
  else:
    return 4


def create_tf_example(metadict, file_path):
  image_path = os.path.join(file_path,metadict['filename'])

  with tf.gfile.GFile(image_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = Image.open(encoded_jpg_io)

  width, height = image.size

  image_filename = metadict['filename'].encode('utf8')
  image_format = b'jpg'

  xmins = []
  xmaxs = []
  ymins = []
  ymaxs = []
  for annotation in metadict['annotations']:
    xmins.append(annotation['xmin'] / width)
    xmaxs.append((annotation['xmin'] + annotation['x_width']) / width)
    ymins.append(annotation['ymin'] / height)
    ymaxs.append((annotation['ymin'] + annotation['y_height']) / height)

  classes_text = [metadict['class'].encode('utf8')]
  classes = [class_text_to_int(metadict['class'])]

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(image_filename),
      'image/source_id': dataset_util.bytes_feature(image_filename),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def dump_tf_record(file_path, yaml_file, tf_record_file_name):
  writer = tf.python_io.TFRecordWriter(tf_record_file_name)
  full_path = os.path.join(file_path,yaml_file)
  with open(full_path, 'r') as yaml_file:
    doc = yaml.load(yaml_file)
    for image_dict in doc:
      tf_example = create_tf_example(image_dict, file_path)
      writer.write(tf_example.SerializeToString())
    writer.close()


def main(_):
  dump_tf_record(REAL_TRAINING_DATA_DIR, REAL_TRAINING_DATA_YAML, 'real_training_data.record')
  dump_tf_record(SIM_TRAINING_DATA_DIR, SIM_TRAINING_DATA_YAML, 'sim_training_data.record')

if __name__ == '__main__':
  tf.app.run()