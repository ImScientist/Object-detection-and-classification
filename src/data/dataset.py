import os
import glob
import logging
import tensorflow as tf

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def parse_ex_proto_fn(ex_proto_serialized):
    """ Parse the input tf.train.Example proto """

    feature_description = {
        'Sugar': tf.io.FixedLenFeature([], tf.string),
        'Gravel': tf.io.FixedLenFeature([], tf.string),
        'Flower': tf.io.FixedLenFeature([], tf.string),
        'Fish': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string)}

    ex_proto = tf.io.parse_single_example(
        serialized=ex_proto_serialized,
        features=feature_description)

    mask = tf.concat([
        tf.io.decode_png(ex_proto['Sugar'], channels=1),
        tf.io.decode_png(ex_proto['Gravel'], channels=1),
        tf.io.decode_png(ex_proto['Flower'], channels=1),
        tf.io.decode_png(ex_proto['Fish'], channels=1),
    ], axis=-1)
    mask = tf.cast(mask, tf.float32)

    image = tf.io.decode_jpeg(ex_proto['image_raw'])

    return {'image': image, 'mask': mask}


def remove_black_pixels_from_masks(el):
    """ Remove the areas in the mask where the image pixels are black """

    non_black_px = tf.clip_by_value(
        tf.reduce_sum(el['image'], axis=-1, keepdims=True),
        tf.constant(0, dtype=tf.uint8),
        tf.constant(1, dtype=tf.uint8))

    non_black_px = tf.cast(non_black_px, dtype=tf.float32)

    mask = el['mask'] * non_black_px

    return {'image': el['image'], 'mask': mask}


def load_dataset(
        data_dir: str,
        batch_size: int,
        prefetch_size: int,
        cache: bool = True,
        cycle_length: int = 2,
        max_files: int = None,
):
    """ Make dataset from a directory with .tfrecords files

    Parameters
    ----------
    data_dir:
    batch_size:
    prefetch_size:
    cache:
    cycle_length: number of files to read concurrently
    max_files: take all files if max_files=None
    """

    files = glob.glob(os.path.join(data_dir, '*.tfrecords'))
    files = files[:max_files]

    ds = (
        tf.data.Dataset
        .from_tensor_slices(files)
        .interleave(
            lambda f: tf.data.TFRecordDataset(f),
            num_parallel_calls=tf.data.AUTOTUNE,
            block_length=batch_size,
            cycle_length=cycle_length)
        .map(parse_ex_proto_fn)
        .map(remove_black_pixels_from_masks)
        .map(lambda x: (tf.cast(x['image'], dtype=tf.float32), x['mask']))
        .batch(batch_size)
        .prefetch(prefetch_size)
    )

    # hm...
    if cache:
        ds = ds.cache()

    # resize_and_rescale_, augment_ = preprocess_layers(new_height=350, new_width=525)

    # # TODO: apply the same transformation to image and mask....
    # if resize_and_rescale:
    #     ds = ds.map(lambda img, mask: (resize_and_rescale_(img), resize_and_rescale_(mask)))

    return ds


def augmentation_fn(
        img, mask, new_height, new_width, training=True
):
    """ Image and mask augmentation """

    seed_args = {'minval': tf.int32.min, 'maxval': tf.int32.max, 'dtype': tf.int32}

    img = img / tf.constant(255, tf.float32)

    img = tf.image.resize_with_pad(img, new_height, new_width)
    mask = tf.image.resize_with_pad(mask, new_height, new_width)

    if training:
        seed = tf.random.uniform((2,), **seed_args)
        img = tf.image.stateless_random_flip_left_right(img, seed)
        mask = tf.image.stateless_random_flip_left_right(mask, seed)

        seed = tf.random.uniform((2,), **seed_args)
        img = tf.image.stateless_random_flip_up_down(img, seed)
        mask = tf.image.stateless_random_flip_up_down(mask, seed)

        # Image transformations that do not require a modification of the mask
        seed = tf.random.uniform((2,), **seed_args)
        img = tf.image.stateless_random_brightness(img, .2, seed)
        img = tf.clip_by_value(img, 0, 1)

        seed = tf.random.uniform((2,), **seed_args)
        img = tf.image.stateless_random_contrast(img, .8, 1.2, seed)
        img = tf.clip_by_value(img, 0, 1)

        seed = tf.random.uniform((2,), **seed_args)
        img = tf.image.stateless_random_saturation(img, 0.8, 1.2, seed)
        img = tf.clip_by_value(img, 0, 1)

        seed = tf.random.uniform((2,), **seed_args)
        img = tf.image.stateless_random_hue(img, 0.2, seed)
        img = tf.clip_by_value(img, 0, 1)

    return img, mask
