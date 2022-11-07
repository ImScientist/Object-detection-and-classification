ds_args = dict(
    batch_size=32,
    prefetch_size=-1,
    cycle_length=4,
    max_files=None,
    take_size=-1)

training_args = dict(
    epochs=100,
    verbose=0)

callbacks_args = dict(
    histogram_freq=0,
    reduce_lr_patience=5,
    profile_batch=(10, 15),
    verbose=0,
    early_stopping_patience=10,
    period=2)
