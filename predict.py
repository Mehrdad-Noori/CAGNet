import os
import cv2
import argparse
import numpy as np
from glob import glob
import tensorflow as tf
from model import BilinearUpsampling
from tensorflow.keras.models import load_model
tf.keras.backend.set_image_data_format('channels_last')


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help='The path to the trained model')
    parser.add_argument("--input_dir", type=str, required=True, default='./data/')
    parser.add_argument("--save_dir", type=str, required=True, default='./save/',
                        help='The path to save the predicted saliency maps')
    return parser.parse_args()


def image_preprocess(image):
    image = image / 255
    image = image.astype(np.float32)
    return image


def read_image(img_dir, dsize=(480,480), mode='bgr'):
    image = cv2.imread(img_dir)
    if mode == 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize)
    image = image_preprocess(image)
    name  = os.path.basename(img_dir)
    return image, name

def write_image(predicted_map, name, save_dir='save'):
    name = os.path.splitext(name)[0] + '.png'
    predicted_map = (predicted_map*255).astype('uint8')
    cv2.imwrite(os.path.join(save_dir, name) , predicted_map)


if __name__ == "__main__":

    cfg = args()
    model = load_model(cfg.model_dir, custom_objects={'BilinearUpsampling': BilinearUpsampling}, compile=False)
    input_height, input_width = model.input.shape[1:3]

    all_images = glob(os.path.join(cfg.data_dir,'*'))
    print('found {} images'.format(len(all_images)))
    for img_dir in all_images:
        image, name = read_image(img_dir, (input_width, input_height))
        pred = model.predict(np.expand_dims(image, 0))
        write_image(pred[0,...,1], name, cfg.save_dir)

