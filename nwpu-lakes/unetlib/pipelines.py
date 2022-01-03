import tensorflow as tf
import numpy as np
import os
import pickle

from .metrics import BinaryMeanIoU
from .preprocessing import make_dataframes_for_flow, make_data_generators


def train_unet(model, img_dir, msk_dir, batch_size=16, epochs=100,
               optimiser='RMSProp', learning_rate=None, callbacks=None,
               save_as=None, save_dir=None):
    f"""Train U-Net model.

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
    callbacks: list of tensorflow.keras.callbacks.Callback, optional
        Callbacks to use during model training. These will be passed
        to the model's `fit` method. By default only a ModelCheckpointer
        callback is used and the output saved according to the `save_as`
        and `save_dir` parameters. Any callbacks provided will replace
        this default bahaviour.
    save_as: str, optional
        Base file name used for saving model history
        and weights. Defaults to "{model.name}_{optimiser_name}_
        lr{learning_rate}_bs{batch_size}e{epochs}".
    save_dir: str, optional
        Directory in which to save output files. Default is current
        directory. If it does not already exist, `save_dir` will be
        created.

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

    # Configure optimiser
    if isinstance(optimiser, str):
        optimiser = tf.keras.optimizers.get(optimiser)
        if learning_rate is not None:
            optimiser.lr = learning_rate

    # Compile model
    model.compile(optimizer=optimiser,
                  loss='binary_crossentropy',
                  metrics=[BinaryMeanIoU(threshold=0.5)]
                  )

    # Output file names
    if save_as is None:
        opt_conf = model.optimizer.get_config()
        o_name = opt_conf['name']
        o_lr = f"{tf.keras.backend.eval(opt_conf['learning_rate']):.3f}"
        save_as = f'{model.name}_{o_name}_lr{o_lr}_bs{batch_size}e{epochs}'

    # Output directory
    if save_dir is None:
        save_dir = '.'
    else:
        os.makedirs(save_dir, exist_ok=True)

    hist_filepath = os.path.join(save_dir, save_as + '.history.pickle')
    weights_filepath = os.path.join(save_dir, save_as + '.weights.h5')

    # Configure callbacks
    if callbacks is None:
        checkpointer = tf.keras.callbacks.ModelCheckpoint(weights_filepath,
                                                          save_best_only=True,
                                                          save_weights_only=True
                                                          )
        callbacks = [checkpointer]

    # Train the model
    history = model.fit(train_gen, epochs=epochs, steps_per_epoch=train_steps,
                        validation_data=val_gen, validation_steps=val_steps,
                        callbacks=callbacks
                        )

    # Save history to pickle
    with open(hist_filepath, 'wb') as f:
        pickle.dump(history.history, f)

    return hist_filepath
