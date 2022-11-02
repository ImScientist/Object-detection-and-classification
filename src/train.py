import os
import re
import logging
import tempfile
import numpy as np
import tensorflow as tf
from src.data.dataset import load_dataset
from src.model import generate_baseline_model
from src import settings
from PIL import Image

tfkc = tf.keras.callbacks

logger = logging.getLogger(__name__)


def log_model_architecture(model, log_dir: str):
    """ Store a non-interactive readable model architecture """

    with tempfile.NamedTemporaryFile('w', suffix=".png") as temp:
        _ = tf.keras.utils.plot_model(
            model,
            to_file=temp.name,
            show_shapes=True,
            dpi=64)

        im_frame = Image.open(temp.name)
        im_frame = np.asarray(im_frame)

        """ Log the figure """
        save_dir = os.path.join(log_dir, 'train')
        file_writer = tf.summary.create_file_writer(save_dir)

        with file_writer.as_default():
            tf.summary.image(
                "model summary",
                tf.constant(im_frame, dtype=tf.uint8)[tf.newaxis, ...],
                step=0)


def create_callbacks(
        log_dir: str,
        save_dir: str = None,
        histogram_freq: int = 0,
        reduce_lr_patience: int = 100,
        profile_batch: tuple = (10, 15),
        verbose: int = 0,
        early_stopping_patience: int = 250,
        period: int = 10
):
    """ Generate model training callbacks """

    callbacks = [
        tfkc.TensorBoard(
            log_dir=log_dir,
            histogram_freq=histogram_freq,
            profile_batch=profile_batch)]

    if reduce_lr_patience is not None:
        callbacks.append(
            tfkc.ReduceLROnPlateau(
                factor=0.2,
                patience=reduce_lr_patience,
                verbose=verbose))

    if early_stopping_patience is not None:
        callbacks.append(
            tfkc.EarlyStopping(patience=early_stopping_patience))

    if save_dir:
        path = os.path.join(
            save_dir,
            'checkpoints',
            'epoch_{epoch:03d}_loss_{val_loss:.4f}_cp.ckpt')

        callbacks.append(
            tfkc.ModelCheckpoint(
                path,
                save_weights_only=True,
                save_best_only=False,
                period=period))

    return callbacks


def get_best_checkpoint(
        checkpoint_dir: str,
        pattern=r'.*_loss_(\d+\.\d{4})_cp.ckpt.index'
):
    """ Parse names of all checkpoints, extract the validation loss
    and return the checkpoint with the lowest loss
    """

    pattern = r'.*_loss_(\d+\.\d{4})_cp.ckpt.index'

    checkpoints = os.listdir(checkpoint_dir)
    checkpoints = map(lambda x: re.fullmatch(pattern, x), checkpoints)
    checkpoints = filter(lambda x: x is not None, checkpoints)
    best_checkpoint = min(checkpoints, key=lambda x: float(x.group(1)))

    checkpoint_name = best_checkpoint.group(0).removesuffix('.index')

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    return checkpoint_path


def get_experiment_id(logs_dir: str):
    """ Generate an unused experiment id by looking at the tensorboard entries """

    experiments = os.listdir(logs_dir)
    experiments = map(lambda x: re.fullmatch(r'ex_(\d{3})', x), experiments)
    experiments = filter(lambda x: x is not None, experiments)
    experiments = map(lambda x: int(x.group(1)), experiments)
    experiments = set(experiments)

    experiment_id = min(set(np.arange(1_000)) - experiments)

    logger.info(f'\n\nExperiment id: {experiment_id}\n\n')

    return experiment_id


def gpu_memory_setup():
    """ Restrict the amount of GPU memory that can be allocated by TensorFlow"""

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=settings.GPU_MEMORY_LIMIT * 1024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


def train(
        ds_dir: str
):
    """ Train and evaluate a model """

    gpu_memory_setup()

    experiment_id = get_experiment_id(settings.TFBOARD_DIR)

    log_dir = os.path.join(settings.TFBOARD_DIR, f'ex_{experiment_id:03d}')
    save_dir = os.path.join(settings.ARTIFACTS_DIR, f'ex_{experiment_id:03d}')

    ds_args = dict(
        batch_size=64,
        prefetch_size=tf.data.AUTOTUNE,
        cache=False,
        cycle_length=4,
        max_files=None)

    training_args = dict(
        epochs=100,
        verbose=0)

    callbacks_args = dict(
        histogram_freq=0,
        reduce_lr_patience=20,
        profile_batch=(10, 15),
        verbose=0,
        early_stopping_patience=250,
        period=2)

    # NO augmentation
    ds_tr = load_dataset(data_dir=os.path.join(ds_dir, 'train'), **ds_args)
    ds_va = load_dataset(data_dir=os.path.join(ds_dir, 'validation'), **ds_args)
    ds_te = load_dataset(data_dir=os.path.join(ds_dir, 'test'), **ds_args)
    ds_sb = load_dataset(data_dir=os.path.join(ds_dir, 'submission'), **ds_args)

    model = generate_baseline_model(height=350, width=525)

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2, from_logits=True))

    # Train and evaluate a model
    callbacks = create_callbacks(log_dir, save_dir, **callbacks_args)

    log_model_architecture(model, log_dir)

    model.fit(ds_tr, validation_data=ds_va, callbacks=callbacks, **training_args)

    # Save the model with the best weights
    checkpoint_path = get_best_checkpoint(os.path.join(save_dir, 'checkpoints'))
    model.load_weights(checkpoint_path)
    model.save(save_dir)
