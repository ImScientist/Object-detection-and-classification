import os
import tensorflow as tf
from src.data.dataset import load_dataset
from src.model import generate_baseline_model


def train(
        ds_dir: str
):
    """ Train and evaluate a model """

    ds_args = dict(
        batch_size=64,
        prefetch_size=tf.data.AUTOTUNE,
        cache=False,
        cycle_length=4,
        max_files=None)

    training_args = dict(
        epochs=100,
        verbose=0)

    ds_tr = load_dataset(data_dir=os.path.join(ds_dir, 'train'), **ds_args)
    ds_va = load_dataset(data_dir=os.path.join(ds_dir, 'validation'), **ds_args)
    ds_te = load_dataset(data_dir=os.path.join(ds_dir, 'test'), **ds_args)

    model = generate_baseline_model(height=350, width=525)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2, from_logits=True))

    model.fit(ds_tr, validation_data=ds_va, **training_args)
