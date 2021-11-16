import tensorflow as tf
import numpy as np
import os
import pickle
import time

from unetlib.metrics import BinaryMeanIoU
from unetlib.preprocessing import make_dataframes_for_flow, make_img_msk_flows


def train_unet(model, nwpu_data_dir, nwpu_mask_dir, batch_size=16, epochs=100,
               optimiser='RMSProp', learning_rate=None, save_as=None):
    """
    Conveniennce wrapper to train a model on the NWPU data.

    Parameters
    ----------
    model
    nwpu_data_dir
    nwpu_mask_dir
    batch_size
    epochs
    optimiser
    learning_rate
    save_as

    Returns
    -------

    """
    # Split the test/train data
    (train_img_df, train_msk_df,
     test_img_df, test_msk_df) = make_dataframes_for_flow(nwpu_data_dir,
                                                          nwpu_mask_dir,
                                                          test_size=0.25,
                                                          random_state=42
                                                          )

    # Split the training data into train and validation generators
    # with augmentation applied to the training data only
    aug_dict = {'rotation_range': 90,
                'horizontal_flip': True,
                'vertical_flip': True,
                'width_shift_range': 0.15,
                'height_shift_range': 0.15,
                'zoom_range': 0.25
                }

    (train_gen, val_gen,
     train_fps, val_fps) = make_img_msk_flows(train_img_df, train_msk_df,
                                              nwpu_data_dir, nwpu_mask_dir,
                                              val_split=0.3, rescale=1 / 255.,
                                              aug_dict=aug_dict,
                                              batch_size=batch_size
                                              )

    # Compute steps per epoch
    train_steps = int(np.ceil(len(train_fps) / batch_size))
    val_steps = int(np.ceil(len(val_fps) / batch_size))

    # Output paths
    if save_as is None:
        hist_filepath = f'{model.name}_bs{batch_size}e{epochs}.history.pickle'
        weights_filepath = f'{model.name}_bs{batch_size}e{epochs}.weights.h5'
    else:
        hist_filepath = f'{save_as}_bs{batch_size}e{epochs}.history.pickle'
        weights_filepath = f'{save_as}_bs{batch_size}e{epochs}.weights.h5'

    if os.path.dirname(hist_filepath) != '':
        os.makedirs(os.path.dirname(hist_filepath), exist_ok=True)
    if os.path.dirname(weights_filepath) != '':
        os.makedirs(os.path.dirname(weights_filepath), exist_ok=True)

    # Configure optimiser
    if isinstance(optimiser, str):
        optimiser = tf.keras.optimizers.get(optimiser)
        if learning_rate is not None:
            optimiser.lr = learning_rate

    # Configure callbacks
    checkpointer = tf.keras.callbacks.ModelCheckpoint(weights_filepath,
                                                      save_best_only=True,
                                                      save_weights_only=True
                                                      )
    callbacks = [checkpointer]

    # Compile model
    model.compile(optimizer=optimiser,
                  loss='binary_crossentropy',
                  metrics=[BinaryMeanIoU(threshold=0.5)]
                  )

    # Train the model and record the time taken
    t1 = time.time()
    history = model.fit(train_gen, epochs=epochs, steps_per_epoch=train_steps,
                        validation_data=val_gen, validation_steps=val_steps,
                        callbacks=callbacks
                        )
    runtime = time.time() - t1

    # Save history to pickle
    with open(hist_filepath, 'wb') as f:
        pickle.dump(history.history, f)

    return hist_filepath, runtime
