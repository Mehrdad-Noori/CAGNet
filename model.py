import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, BatchNormalization, \
    Activation, multiply, add, Concatenate, Reshape, InputSpec, Layer
from tensorflow.keras.models import Model, load_model
from tensorflow.python.keras.utils import conv_utils

tf.keras.backend.set_image_data_format('channels_last')


def cagnet_model(backbone='VGG16', input_shape=(480, 480, 3), mfem_filters=8, rrm_filters=32,
           backbone_weights='imagenet', load_model_dir=None):
    """
    Builds CAGNet model

    Parameters
    ----------
    backbone : str
        The backbone model. one of the following options: VGG16, ResNet50, NASNetMobile, NASNetLarge"
    input_shape : tuple
        Shape of the input data (height, width, channel)
    mfem_filters : int
        The number of filters used in Multi-scale Feature Extraction Module (MFEM). The default is 8.
    rrm_filters : int
        The number of filters used in Residual Refinement Module (RRM). The default is 32.
    backbone_weights : str
         The initialization type of backbone model. One of `None` (random initialization), 'imagenet'
         (pre-training on ImageNet). The default is 'imagenet'.
    saved_model_dir : str
        If spesified, the model weights will be loaded from this path. The default is None.

    Returns
    -------
    model : tf.keras.model
        The created keras model with respect to the input parameters
    """

    if load_model_dir:
        model = load_model(load_model_dir)
        print('CAGNet model loaded!')
        return model

    ## Feature Extraction Network
    if backbone == 'VGG16':
        backbone_model, backbone_levels = vgg16(input_shape, backbone_weights)
    elif backbone == 'ResNet50':
        backbone_model, backbone_levels = resnet50(input_shape, backbone_weights)
    elif backbone == 'NASNetMobile':
        backbone_model, backbone_levels = nasnet_mobile(input_shape, backbone_weights)
    elif backbone == 'NASNetLarge':
        backbone_model, backbone_levels = nasnet_large(input_shape, backbone_weights)
    else:
        raise ValueError(
            "The name of the Backbone_model must be one of the following options: VGG16, ResNet50, NASNetMobile, NASNetLarge")

    # feature map shapes for different levels
    # level_shapes = [(15, 15), (30, 30), (60, 60), (120, 120)] => levels d, c, b, a
    level_shapes = [value.shape[1:3] for _, value in backbone_levels.items()]

    level_d = feature_extraction_module(backbone_levels['d'], mfem_filters)
    # upsample level d to level c, b and a
    level_d_upsamples = [BilinearUpsampling(output_size=size)(level_d) for size in
                         level_shapes[1:]]

    level_c = feature_extraction_module(backbone_levels['c'], mfem_filters)
    # upsample level c to level b and a
    level_c_upsamples = [BilinearUpsampling(output_size=size)(level_c) for size in
                         level_shapes[2:]]

    level_b = feature_extraction_module(backbone_levels['b'], mfem_filters)
    # upsample level b to level a
    level_b_upsamples = [BilinearUpsampling(output_size=size)(level_c) for size in level_shapes[3:]]

    level_a = feature_extraction_module(backbone_levels['a'], mfem_filters)

    ## Feature Guide Network
    guided_dc = guide_module(level_d_upsamples[0], level_c)
    guided_db = guide_module(level_d_upsamples[1], level_b)
    guided_da = guide_module(level_d_upsamples[2], level_a)

    guided_cb = guide_module(level_c_upsamples[0], level_b)
    guided_ca = guide_module(level_c_upsamples[1], level_a)

    guided_ba = guide_module(level_b_upsamples[0], level_a)

    ## Feature Fusion Network
    added_dc = guided_dc
    added_db_cb = add([guided_db, guided_cb])
    added_da_ca_ba = add([guided_da, guided_ca, guided_ba])

    final_level_c = residual_refinement_module(added_dc, rrm_filters)
    final_level_b = residual_refinement_module(added_db_cb, rrm_filters)
    final_level_a = residual_refinement_module(added_da_ca_ba, rrm_filters)

    z = BilinearUpsampling(output_size=(60, 60))(final_level_c)
    z = add([z, final_level_b])
    z = residual_refinement_module(z, rrm_filters)
    z = BilinearUpsampling(output_size=(120, 120))(z)
    z = add([z, final_level_a])
    z = residual_refinement_module(z, rrm_filters)
    z = BilinearUpsampling(output_size=(240, 240))(z)
    z = residual_refinement_module(z, rrm_filters)
    z = BilinearUpsampling(output_size=(480, 480))(z)

    z = Conv2D(2, (1, 1), padding='same', kernel_initializer='he_normal')(z)
    z = Activation('softmax')(z)
    model = Model(backbone_model.input, z)

    print('CAGNet model created!')
    print('{:20} {}'.format('backbone model:', backbone))
    print('{:20} {}'.format('input shape:', input_shape))
    print('{:20} {}'.format('encoder parameters:', backbone_model.count_params()))
    print('{:20} {}'.format('decoder parameters:', model.count_params() - backbone_model.count_params()))
    print('{:20} {}'.format('total parameters:', model.count_params()))

    return model


def nasnet_large(input_shape, weights):
    backbone_model = applications.nasnet.NASNetLarge(input_shape=input_shape, weights=weights, include_top=False)
    backbone_levels = {'d': backbone_model.output,
                       'c': backbone_model.get_layer("normal_concat_12").output,
                       'b': backbone_model.get_layer("normal_concat_5").output,
                       'a': backbone_model.get_layer("reduction_concat_stem_1").output}
    return backbone_model, backbone_levels


def nasnet_mobile(input_shape, weights):
    backbone_model = applications.nasnet.NASNetMobile(input_shape=input_shape, include_top=False, weights=weights)
    backbone_levels = {'d': backbone_model.output,
                       'c': backbone_model.get_layer("normal_concat_8").output,
                       'b': backbone_model.get_layer("normal_concat_3").output,
                       'a': backbone_model.get_layer("reduction_concat_stem_1").output}
    return backbone_model, backbone_levels


def resnet50(input_shape, weights):
    backbone_model = applications.resnet50.ResNet50(input_shape=input_shape, include_top=False, weights=weights)
    backbone_levels = {'d': backbone_model.output,
                       'c': backbone_model.get_layer("conv4_block6_out").output,
                       'b': backbone_model.get_layer("conv3_block4_out").output,
                       'a': backbone_model.get_layer("conv2_block3_out").output}
    return backbone_model, backbone_levels


def vgg16(input_shape, weights):
    backbone_model = applications.vgg16.VGG16(input_shape=input_shape, include_top=False, weights=weights)
    added_layer = Conv2D(1024, (3, 3), activation="relu", padding="same")(backbone_model.output)
    backbone_levels = {'d': added_layer,
                       'c': backbone_model.get_layer("block5_conv3").output,
                       'b': backbone_model.get_layer("block4_conv3").output,
                       'a': backbone_model.get_layer("block3_conv3").output}
    return backbone_model, backbone_levels


def global_conv_network(input_tensor, filters, k):
    """
    Creates the Global Convolutional Network (GCN)
    article: https://arxiv.org/abs/1703.02719

    Parameters
    ----------
    input_tensor : tf.Tensor
    filters : int
        The number of filters used in GCN Conv layers.
    k : int
        The kernel size.

    Returns
    -------
    output_tensor : tf.Tensor
    """
    x1 = Conv2D(filters, (1, k), strides=(1, 1), padding='same', kernel_initializer='he_normal')(input_tensor)
    x1 = Conv2D(filters, (k, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x1)
    x2 = Conv2D(filters, (k, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(input_tensor)
    x2 = Conv2D(filters, (1, k), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x2)
    output_tensor = add([x1, x2])
    output_tensor = BatchNormalization(epsilon=1e-5)(output_tensor)
    output_tensor = Activation('relu')(output_tensor)

    return output_tensor


def feature_extraction_module(input_tensor, filters):
    """
    Creates the Multi-scale Feature Extraction Module (MFEM).

    Parameters
    ----------
    input_tensor : tf.Tensor
    filters : int

    Returns
    -------
    output_tensor : tf.Tensor
    """

    path1 = Conv2D(filters, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(input_tensor)
    path1 = BatchNormalization(epsilon=1e-5)(path1)
    path1 = Activation('relu')(path1)

    path2 = global_conv_network(input_tensor, filters, 7)
    path3 = global_conv_network(input_tensor, filters, 11)
    path4 = global_conv_network(input_tensor, filters, 15)
    output_tensor = Concatenate()([path1, path2, path3, path4])

    return output_tensor


def guide_module(highlevel_input, lowlevel_input):
    """
    Creates the Guild Module (GM). This module consists of High-level Guide and Low-level
    Guide branches and is adopted to guide the features of the different levels.

    Parameters
    ----------
    highlevel_input : tf.Tensor
    lowlevel_input : tf.Tensor

    Returns
    -------
    output_tensor : tf.Tensor

    """

    low_channels = lowlevel_input.shape[-1]
    high_channels = highlevel_input.shape[-1]
    concat = Concatenate()([highlevel_input, lowlevel_input])
    # Low-level Guide Branch
    low_branch = GlobalAveragePooling2D()(concat)
    low_branch = Reshape((1, 1, int(high_channels + low_channels)))(low_branch)
    low_branch = Conv2D(int(high_channels + low_channels), (1, 1), activation='relu', use_bias=False,
                        kernel_initializer='he_normal')(low_branch)
    low_branch = Conv2D(low_channels, (1, 1), activation='sigmoid', use_bias=False)(low_branch)
    guided_low = multiply([lowlevel_input, low_branch])
    # High-level Guide Branch
    high_branch = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False)(concat)
    guided_high = multiply([highlevel_input, high_branch])
    output_tensor = add([guided_high, guided_low])
    return output_tensor


def residual_refinement_module(input_tensor, filters, strides=(1, 1)):
    """
    Creates the Residual Refinement Module (RRM). RRM is a residual block with spatial
    attention and is adopted to refine the features effectively.

    Parameters
    ----------
    input_tensor : tf.Tensor
    filters : int
    strides : tuple

    Returns
    -------
    output_tensor : tf.Tensor
    """

    x = BatchNormalization(epsilon=1e-5)(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    x = spatial_attention(x)
    output_tensor = add([x, input_tensor])

    return output_tensor


def spatial_attention(input_tensor):
    """
    Spatial Attention

    Parameters
    ----------
    input_tensor : tf.Tensor

    Returns
    -------
    output_tensor : tf.Tensor
    """
    x = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False)(input_tensor)
    output_tensor = multiply([input_tensor, x])
    return output_tensor


class BilinearUpsampling(Layer):
    """
    Creates a bilinear upsampling layer.

    Parameters
    ----------
    upsampling: int
        the upsampling factors for rows and columns.
    output_size: tuple
        use this arg instead of upsampling arg if your desired size is not an integer factor of the input size
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                     input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                    input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return tf.compat.v1.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                               inputs.shape[2] * self.upsampling[1]),
                                                      align_corners=True)
        else:
            return tf.compat.v1.image.resize_bilinear(inputs, (self.output_size[0],
                                                               self.output_size[1]),
                                                      align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
