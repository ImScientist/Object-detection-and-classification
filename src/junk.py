import logging
import tensorflow as tf

tfkl = tf.keras.layers

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# I will not be able to apply the same random trf to both image and mask?
# They have a seed value
def preprocess_layers(new_height, new_width):
    resize_and_rescale = tf.keras.Sequential([
        tfkl.Resizing(height=new_height, width=new_width),
        tfkl.Rescaling(1. / 255)
    ])

    augment = tf.keras.Sequential([
        tfkl.RandomFlip("horizontal_and_vertical"),
        tfkl.RandomRotation(0.1, fill_mode='constant')
    ])

    # aug_ds = train_ds.map(lambda x, y: (augment(x, training=True), y))

    return resize_and_rescale, augment


def preprocess_img():
    """ Random transformations that can be applied to the image and mask """

    tf.image.flip_left_right()
    tf.image.flip_up_down()
    tf.image.adjust_saturation


# It does not work:
# All layers in a Sequential model should have a single output tensor.
# For multi-output layers, use the functional API.
class ExtendedRandomFlip(tfkl.Layer):
    """ Extension of the tfkl.RandomFlip """

    def __init__(self, p=.5, **kwargs):
        super().__init__(**kwargs)
        self.p = tf.constant(p, dtype=tf.float32)

    def call(self, inputs, training=True):
        if training:
            return self._random_flip(inputs)
        else:
            return inputs

    def _random_flip(self, inputs):

        if tf.random.uniform([]) < self.p:

            seed = tf.random.uniform(
                shape=(2,), minval=tf.int32.min, maxval=tf.int32.max,
                dtype=tf.int32)

            image = tf.image.stateless_random_flip_left_right(
                inputs['image'], seed=seed)
            mask = tf.image.stateless_random_flip_left_right(
                inputs['mask'], seed=seed)

            outputs = {'image': image, 'mask': mask}
        else:
            outputs = inputs

        return outputs





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
