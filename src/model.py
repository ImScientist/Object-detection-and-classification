import logging
import tensorflow as tf

tfkl = tf.keras.layers

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# TODO: If I want to use them I have to create a custom class with modified
#  train step...
def augmentation_layers(height, width, new_height, new_width):
    """ Image and mask augmentation layers

    There seems to be a bug in tf 2.9 and 2.10 that does not allow us to use
    the RandomXXX layers in an efficient way:
        https://github.com/keras-team/keras-cv/issues/581
    """

    img_in = tfkl.Input(shape=(height, width, 3), dtype=tf.float32)
    mask_in = tfkl.Input(shape=(height, width, 4), dtype=tf.float32)

    x = tfkl.concatenate([img_in, mask_in], axis=-1)
    x = tfkl.Resizing(height=new_height, width=new_width)(x)
    x = tfkl.RandomFlip("horizontal_and_vertical")(x)
    x = tfkl.RandomRotation(.07, fill_mode='constant')(x)

    img = tfkl.Lambda(lambda t: t[..., :3])(x)
    mask = tfkl.Lambda(lambda t: t[..., 3:])(x)

    # Image transformations that do not require a modification of the mask
    # img = tfkl.RandomBrightness(factor=.1)(img)
    # img = tfkl.RandomContrast(factor=.1)(img)

    img = tfkl.Lambda(lambda t: t / tf.constant(255, tf.float32))(img)

    model = tf.keras.Model(inputs=[img_in, mask_in], outputs=[img, mask])

    return model


# The simplest possible model....
def generate_baseline_model(height, width):
    """ Baseline model """

    args = {'activation': 'relu', 'padding': "same"}

    model = tf.keras.Sequential([
        tfkl.Conv2D(128, [3, 3], input_shape=(height, width, 3), **args),
        tfkl.Conv2D(128, [3, 3], **args),
        tfkl.Conv2D(64, [3, 3], **args),
        tfkl.Conv2D(32, [3, 3], **args),
        tfkl.Conv2D(4, [3, 3], **args)
    ])

    return model
