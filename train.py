import os
import argparse
import numpy as np
from loss import custom_loss
from model import cagnet_model
from data_generator import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint, TensorBoard


def args():
    parser = argparse.ArgumentParser(description="Source code for training CAGNet models")
    parser.add_argument("--backbone_model", type=str, default='VGG16', choices=['VGG16', 'ResNet50', 'NASNetMobile',
                                                                                'NASNetLarge'])
    parser.add_argument("--backbone_weights", type=str, default='imagenet', choices=['imagenet', 'scratch'])
    parser.add_argument("--input_shape", type=tuple, default=(480, 480, 3))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=8e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train_dir", type=str, default='./data/')
    parser.add_argument("--save_dir", type=str, default='./save/')
    parser.add_argument("--use_multiprocessing", type=bool, default=True)
    parser.add_argument("--load_model", type=str, default=None, help='If specified, before training, the model '
                                                                     'weights will be loaded from this path otherwise '
                                                                     ' the model will be trained from scratch.')

    return parser.parse_args()


def image_preprocess(image):
    image = image / 255
    image = image.astype(np.float32)
    return image


def mask_preprocess(mask):
    label = mask / 255
    label = (label >= 0.5).astype(np.bool)
    label = to_categorical(label, num_classes=2)
    return label


if __name__ == "__main__":

    cfg = args()
    image_datagen_train = ImageDataGenerator(rotation_range=12, horizontal_flip=True,
                                             preprocessing_function=image_preprocess)
    mask_datagen_train = ImageDataGenerator(n_categorical=2, rotation_range=12, horizontal_flip=True,
                                            preprocessing_function=mask_preprocess)
    image_generator_train = image_datagen_train.flow_from_directory(os.path.join(cfg.train_dir, 'images'),
                                                                    target_size=cfg.input_shape[:-1], color_mode='rgb',
                                                                    class_mode=None, seed=1, batch_size=cfg.batch_size)
    mask_generator_train = mask_datagen_train.flow_from_directory(os.path.join(cfg.train_dir, 'masks'),
                                                                  target_size=cfg.input_shape[:-1], color_mode="grayscale",
                                                                  class_mode=None, seed=1, batch_size=cfg.batch_size)

    train_generator = zip(image_generator_train, mask_generator_train)
    steps_per_epoch = len(image_generator_train)

    if cfg.backbone_weights == 'scratch':
        cfg.backbone_weights = None

    model = cagnet_model(cfg.backbone_model, cfg.input_shape, backbone_weights=cfg.backbone_weights, load_model_dir=cfg.load_model)
    model.compile(optimizer=SGD(lr=cfg.learning_rate, momentum=0.9), loss=custom_loss, metrics=["accuracy"])

    if not os.path.isdir(cfg.save_dir):
        os.mkdir(cfg.save_dir)

    logger = CSVLogger(os.path.join(cfg.save_dir, 'log.txt'), append=True)
    # If the training loss does not decrease for 10 epochs, the learning rate is divided by 10.
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, mode='min', min_delta=0.0001)
    check_point = ModelCheckpoint(filepath=os.path.join(cfg.save_dir, 'cagnet_{epoch:03d}_{loss:.4f}.hdf5'),
                                  monitor='loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard(os.path.join(cfg.save_dir, 'tensorboard'))
    callbacks = [logger, reduce_lr, check_point, tensorboard]

    model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=cfg.epochs,
                        use_multiprocessing=cfg.use_multiprocessing, verbose=1, callbacks=callbacks, shuffle=True)
