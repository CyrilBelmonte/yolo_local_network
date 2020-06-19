import threading
import time
import os
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from myTools import network_tools as nt

from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output/local_cnn/output_' + str(int(time.time())) + '.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_string('ip', '127.0.0.1', 'ip by default')
flags.DEFINE_integer('port', 12346, 'port default')


def main(_argv):
    id_img = 0
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    logging.info('initialization connexion at {}:{}'.format(FLAGS.ip, str(FLAGS.port)))
    connexion_1 = time.time()
    sk = nt.init_connexion(FLAGS.ip, FLAGS.port)
    connexion_2 = time.time()
    connexion = connexion_2 - connexion_1
    logging.info("connected to {} port {} in {:.3f}ms".format(FLAGS.ip, FLAGS.port, connexion))

    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
    else:
        img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, FLAGS.size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    nt.get_more_data(img, (boxes, scores, classes, nums), class_names)

    logging.info('time: {}'.format(t2 - t1))

    logging.info('primary detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {:.2f}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i])))

    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)

    cv2.imwrite(FLAGS.output, img)

    logging.info('output saved to: {}'.format(FLAGS.output))
    sk.close()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
