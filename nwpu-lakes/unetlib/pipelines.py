import tensorflow as tf
import numpy as np
import os
import pickle
import time

from .metrics import BinaryMeanIoU
from .preprocessing import make_dataframes_for_flow, make_data_generators


# Wrapper for training process with some hyperparameters
def train_unet(model, img_dir, msk_dir, batch_size=16, epochs=100,
               optimiser='RMSProp', learning_rate=None, save_as=None):
    """Train U-Net model.

    Train `model` and save history and best weights.

    Parameters
    ----------
    model: tensorflow.keras.models.Model
        UNet model that will be trained.
    img_dir, msk_dir: str
        Paths to image and mask directories.
    batch_size: int, default=16
        Number of image/mask pairs in each batch of training
        and validation data.
    epochs: int, default=100
        Duration of training i.e. number of passes through the
        full training set.
    optimiser: str, default='RMSProp'
        Optimisation algorithm. Refer to Keras documentation
        for available optimisers.
    learning_rate: float, optional
        Learning rate for `optimiser`. If None, will use
        `optimiser`'s default learning rate'.
    save_as: str, optional
        Base output path name used for savinng model history
        and weights. Defaults to `model.name`. Note that the
        actual output file names are laundered to include the
        `batch_size` and `epochs` arguments.

    Returns
    -------
    str
        Path to model history output file.
    """
    # Split the test/train data
    (train_img_df, train_msk_df,
     test_img_df, test_msk_df) = make_dataframes_for_flow(img_dir,
                                                          msk_dir,
                                                          test_split=0.25,
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
     train_fps, val_fps) = make_data_generators(train_img_df, train_msk_df,
                                                img_dir, msk_dir,
                                                val_split=0.3,
                                                batch_size=batch_size,
                                                aug_dict=aug_dict,
                                                aug_seed=42)

    # Compute steps per epoch
    train_steps = int(np.ceil(len(train_fps) / batch_size))
    val_steps = int(np.ceil(len(val_fps) / batch_size))

    # Output paths
    if save_as is None:
        base_name = f'{model.name}_bs{batch_size}e{epochs}'
    else:
        base_name = f'{save_as}_bs{batch_size}e{epochs}'

    hist_filepath = base_name + '.history.pickle'
    weights_filepath = base_name + '.weights.h5'

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
    print(f"Model {base_name} training time: {time.time() - t1}")

    # Save history to pickle
    with open(hist_filepath, 'wb') as f:
        pickle.dump(history.history, f)

    return hist_filepath
