import numpy as np

import threading
import myTools.image_tools as img_t

import socket, select
import sys
from absl import app, flags, logging

np.set_printoptions(threshold=sys.maxsize)


def init_connexion(host="127.0.0.1", port=12346):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # connect to server on local computer
    s.connect((host, port))
    # message you send to server
    message = "0;YOLO;CAT;DATA;"
    s.send(message.encode('utf-8'))
    data = s.recv(3000000)
    while data.decode('utf-8') != "OK;":
        data = s.recv(3000000)
    # logging.info("\tConnexion ok")
    return s


def get_more_data(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    process_s = []
    nb_image = 0
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))

        if class_names[int(classes[i])] == "cat" or class_names[int(classes[i])] == "dog":
            x1, y1 = x1y1
            x2, y2 = x2y2
            img_crop = img[y1:y1 + (y2 - y1), x1:x1 + (x2 - x1)]
            nb_image = nb_image + 1
            img_to_net = img_t.load_img_opencv(img_crop)
            if class_names[int(classes[i])] == "cat":
                process = ThreadSend("cat", img_to_net)
                process.setDaemon(True)
                process.start()
                process_s.append(process)
            else:
                process = ThreadSend("dog", img_to_net)
                process.setDaemon(True)
                process.start()
                process_s.append(process)
    return nb_image


def send_data(cat, data, s):
    message = '1;YOLO;{};{};'.format(cat.upper(), np.array2string(data, separator=","))
    logging.info(
        'Send image category : {} message size = {:.2f}Mb'.format(cat.upper(), (sys.getsizeof(message) / 1000000)))
    s.send(message.encode('utf-8'))
    s.close()


class ThreadSend(threading.Thread):
    def __init__(self, cat, data):
        threading.Thread.__init__(self)
        self.cat = cat
        self.data = data

    def run(self):
        host = "127.0.0.1"
        port = 12346
        sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sk.connect((host, port))
        sk.setblocking(False)
        send_data(self.cat, self.data, sk)
