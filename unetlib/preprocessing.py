import numpy as np
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_lakes_with_masks(img_dir, mask_dir, return_masks=True, img_ext='jpg', mask_ext='png'):
    """ Determine lake images that have corresponding masks.
    
    Valid images are those that have a corresponding mask in
    mask_dir.
    
    """
    img_fns = sorted([fn for fn in os.listdir(img_dir) if fn.endswith(img_ext)]) 
    msk_fns = sorted([fn for fn in os.listdir(mask_dir) if fn.endswith(mask_ext)])
    
    img_fns = sorted(filter(lambda fn: f'{os.path.splitext(fn)[0]}_mask.png' in msk_fns,
                            img_fns)
                    )
    if return_masks:
        result = np.array(img_fns), np.array(msk_fns)
    else:
        result = np.array(img_fns)
    
    return result


def make_dataframes_for_flow(img_dir, msk_dir, test_size=0.0,
                             random_state=None):
    """
    """
    valid_images, valid_masks = get_lakes_with_masks(img_dir, msk_dir)
    
    img_df = pd.DataFrame({'filename':valid_images})
    msk_df = pd.DataFrame({'filename':valid_masks})
    
    test_idxs = None
    if test_size == 0:
        return (img_df, msk_df)
        
    else:
        # Get train/test indices
        test_samples = int(img_df.shape[0] * test_size)
        test_idxs = img_df.sample(test_samples, random_state=random_state).index
        train_idxs = img_df.index.difference(test_idxs)
        
        # subset the images and mask dataframes
        train_img_df = img_df.iloc[train_idxs]
        train_msk_df = msk_df.iloc[train_idxs]
        
        test_img_df = img_df.iloc[test_idxs]
        test_msk_df = msk_df.iloc[test_idxs]
        
        return (train_img_df, train_msk_df, test_img_df, test_msk_df)
    
    
# Function to produce the train and val generators
def make_img_datagen(img_df, msk_df, img_dir, msk_dir, val_split=0.0,
                     batch_size=32, **kwargs):
    """Create image & mask generators
    
    Parameters
    ----------
    img_df: pd.DataFrame
    msk_df: pd.DataFrame
    img_dir: str
    msk_dir: str
    val_split: float, optional
    batch_size: int, optional
    **kwargs: dict
        Other keyword arguments that will be passed to the
        ImageDataGenerator
        
    Returns
    -------
    generator:
        If val_split is 0, only the training data generator
        is returned.
    tuple of generators and lists:
        If val_split > 0, training and validation generators
        are returned along with the filenames used for each
        split.
    """
    # Instantiate the data generator specifying a validation split.
    datagen = ImageDataGenerator(validation_split=val_split, **kwargs)
    
    # Create the training data generators first
    train_img_gen = datagen.flow_from_dataframe(img_df,
                                                img_dir,
                                                y_col=None,
                                                class_mode=None,
                                                shuffle=None,
                                                subset='training',
                                                batch_size=batch_size
                                               )
    
    train_msk_gen = datagen.flow_from_dataframe(msk_df,
                                                msk_dir,
                                                y_col=None,
                                                class_mode=None,
                                                shuffle=None,
                                                subset='training',
                                                color_mode='grayscale',
                                                batch_size=batch_size
                                               )
    # Combine
    train_gen = (pair for pair in zip(train_img_gen, train_msk_gen))
    
    # If val_split is greater than 0, i.e. the data is actually split
    # create the validation data generators
    val_gen = None
    if val_split > 0:
        val_img_gen = datagen.flow_from_dataframe(img_df,
                                                  img_dir,
                                                  y_col=None,
                                                  class_mode=None,
                                                  shuffle=None,
                                                  subset='validation',
                                                  batch_size=batch_size
                                                 )

        val_msk_gen = datagen.flow_from_dataframe(msk_df,
                                                  msk_dir,
                                                  y_col=None,
                                                  class_mode=None,
                                                  shuffle=None,
                                                  subset='validation',
                                                  color_mode='grayscale',
                                                  batch_size=batch_size
                                                 )
        
        val_gen = (pair for pair in zip(val_img_gen, val_msk_gen))
        
    if val_gen is None:
        # If no split has been done just return train_gen
        return train_gen
    else:
        # Otherwise return train_gen and val_gen and also
        # the file paths that fall into each split to ensure
        # reproducibility
        
        train_fps = train_img_gen.filepaths
        val_fps = val_img_gen.filepaths
        
        return (train_gen, val_gen, train_fps, val_fps)