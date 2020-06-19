import time
import threading
import cv2
import tensorflow as tf

from absl import app, flags, logging
from absl.flags import FLAGS

from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
from myTools import network_tools as nt

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
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

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    logging.info('initialization connexion at {}:{}'.format(FLAGS.ip, str(FLAGS.port)))
    connexion_1 = time.time()
    sk = nt.init_connexion(FLAGS.ip, FLAGS.port)
    sk.close()
    connexion_2 = time.time()
    connexion = connexion_2 - connexion_1
    logging.info("connected to {} port {} in {:.3f}ms".format(FLAGS.ip, FLAGS.port, connexion))

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        id_img = id_img + 1
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        t2 = time.time()

        process = ThreadSending(img, (boxes, scores, classes, nums), class_names, id_img)
        process.setDaemon(True)
        process.start()
        times.append(t2 - t1)

        times = times[-20:]

        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)

        img = cv2.putText(img, "Time prim: {:.2f}ms".format(sum(times) / len(times) * 1000), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

        if FLAGS.output:
            out.write(img)
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


class ThreadSending(threading.Thread):
    def __init__(self, img, output, class_names, id_img):
        threading.Thread.__init__(self)
        self.img = img
        self.output = output
        self.class_names = class_names
        self.id_img = id_img

    def run(self):
        nt.get_more_data(self.img, self.output, self.class_names)

        # logging.info("image id{} in {:.3f} ".format(self.id_img, (send_t_2 - send_t_1)))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
