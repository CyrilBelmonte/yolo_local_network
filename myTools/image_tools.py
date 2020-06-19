import numpy as np
from skimage import transform
from keras_preprocessing import image


def load_img_opencv(img, width=224, height=224):
    np_image = img
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (width, height, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


def load_img_opencv2(img, width=224, height=224):
    np_image = img
    np_image = np.array(np_image).astype('uint8')
    np_image = transform.resize(np_image, (width, height, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


def load_img_keras(file, width=224, height=224):
    img_width, img_height = width, height
    img = image.load_img(file, target_size=(img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img
