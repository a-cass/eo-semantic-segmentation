import os

import numpy as np
import pandas as pd
from skimage.draw import polygon
from skimage.io import imread, imsave
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def labels2mask(img_path, labels_dict, out_path=None, out_ext=None,
                mask_val=255, dtype=np.uint8, overwrite=True, **kwargs):
    """Convert label coordinates to label image.

    The coordinates must be in units of image pixels.

    Parameters
    ----------
    img_path : str
        Path to input image.
    labels_dict : dict
        Dictionary of image labels. Top level keys
        should be image file names and values should
        be dictionaries containing image label information
        under the key 'regions'.
    out_path : str, optional
        Path to save mask image. By default the mask image
        is saved to the same direction as `img_path` with
        the suffix '_mask'.
    out_ext : str, optional
        Output image extension e.g. 'png', 'jpg'. By default
        this will match the input image.
    mask_val : int, default=255
        Pixel value to assign to masked areas.
    dtype : subtype of numpy.number, default=numpy.uint8
        Output image data type e.g. numpy.uint8, numpy.float.
    overwrite : bool, default=True
        Overwrite 'out_path' if it already exists.
    **kwargs : dict, optional
        Keyword arguments passed to `skimage.io.imsave`.
        Refer to the `skimage.io.imsave` documentation for
        a list of all possible arguments.

    Returns
    -------
    str
        Path to output image file. Same as `out_path` if it
        is not None.
    """
    # launder output file path
    if out_path is None:
        out_path = '_mask'.join(os.path.splitext(img_path))

    if out_ext:
        out_path = os.path.splitext(out_path)[0] + f'.{out_ext.lower()}'

    # handle overwrite
    if os.path.exists(out_path):
        if not overwrite:
            return out_path

    # get water polygon info
    img_name = os.path.basename(img_path)
    regions = labels_dict[img_name]['regions']

    # create mask array
    # output is only 2D as opposed to the 3 channel input image
    mask_shp = imread(img_path).shape[:2]
    mask_arr = np.zeros(mask_shp, dtype=dtype)

    # Cycle through regions and set corresponsing pixels to mask_val
    for region_num, region_shape in regions.items():
        x = region_shape['shape_attributes']['all_points_x']
        y = region_shape['shape_attributes']['all_points_y']

        # The skimage polygon function requires coordinates as rows, cols
        r, c = polygon(y, x, mask_shp)
        mask_arr[r, c] = mask_val

    # write to new image
    imsave(out_path, mask_arr, check_contrast=False, **kwargs)

    return out_path


def get_lakes_with_masks(img_dir, msk_dir, return_masks=True, img_ext='jpg',
                         msk_ext='png'):
    """Determine lake images that have corresponding masks.

    Valid images are those that have a corresponding mask in
    mask_dir.

    Parameters
    ----------
    img_dir, msk_dir: str
        Paths to image and mask directories.
    return_masks: bool, default=True
        If `True` also return list of mask file names.
    img_ext: str, default='jpg'
        Extension of image files. Used for file selection.
    msk_ext: str, default='png'
        Extension of mask files. Used for file selection.

    Returns
    -------
    tuple of lists or list
        List(s) of file names for valid images and their
        masks. If `return_masks` is `False`, only a single
        list of image file names is returned.
    """
    img_fns = sorted([fn for fn in os.listdir(img_dir) if fn.endswith(img_ext)]) 
    msk_fns = sorted([fn for fn in os.listdir(msk_dir) if fn.endswith(msk_ext)])
    
    img_fns = sorted(filter(lambda fn: f'{os.path.splitext(fn)[0]}_mask.png' in msk_fns,
                            img_fns)
                    )
    if return_masks:
        result = np.array(img_fns), np.array(msk_fns)
    else:
        result = np.array(img_fns)
    
    return result


def make_dataframes_for_flow(img_dir, msk_dir, test_split=0.0,
                             random_state=None):
    """Generate image and mask dataframes

    Parameters
    ----------
    img_dir, msk_dir: str
        Paths to image and mask directories.
    test_split: float, default=0.0
        Proportion of images to reserve for the test set.
    random_state: int, optional
        Seed for reproducibility. By default no
        seed is set and results will vary between
        runs.

    Returns
    -------
    tuple of pandas.core.frame.DataFrame
        If test size is 0 just the image and masks
        DataFrames are returned. Otherwise the data
        is split and this function will return, in
        this order, DataFrames for the training images,
        training masks, testing images, testing masks.
    """
    valid_images, valid_masks = get_lakes_with_masks(img_dir, msk_dir)
    
    img_df = pd.DataFrame({'filename': valid_images})
    msk_df = pd.DataFrame({'filename': valid_masks})

    if test_split == 0:
        return img_df, msk_df
        
    else:
        # Get train/test indices
        test_samples = int(img_df.shape[0] * test_split)
        test_idxs = img_df.sample(test_samples, random_state=random_state).index
        train_idxs = img_df.index.difference(test_idxs)
        
        # subset the images and mask dataframes
        train_img_df = img_df.iloc[train_idxs]
        train_msk_df = msk_df.iloc[train_idxs]
        
        test_img_df = img_df.iloc[test_idxs]
        test_msk_df = msk_df.iloc[test_idxs]
        
        return train_img_df, train_msk_df, test_img_df, test_msk_df


def make_data_generators(img_df, msk_df, img_dir, msk_dir, val_split=0.0,
                         batch_size=32, rescale=1 / 255, aug_dict=None,
                         aug_seed=None):
    """Create image & mask data generators.

    Optionally augmentation can be applied. Note that
    augmentations apply only to the training data not
    the validation data.

    The ordering of files in `img_df` and `msk_df` must match.
    For example, the first row of `msk_df` should correspond
    to the mask for the first image in `img_df`.

    If this is not the case the generators will fail to generate
    flows of matching image and mask pairs.

    Parameters
    ----------
    img_df, msk_df: pandas.core.frame.DataFrame
        DataFrames containing image and mask file names respectively.
        These must be stored in a `filename` column.
    img_dir, msk_dir: str
        Paths to image and mask directories.
    val_split: float, default=0.0
        Proportion of images to reserve for the validation set.
    batch_size: int, default=32
        Number of image/mask pairs in each batch.
    rescale: float, default=1/255
        Scaling to apply to image and mask data. Defaults to
        1/255 for rescaling 8bit unit data.
    aug_dict: dict, optional
        Keyword argument dictionary for augmentations.
        See `ImageDataGenerator` for valid keywords.
        Defaults to None resulting in no augmentations.
    aug_seed: int, optional
        Random seed for augmentations. Must be provided if
        use augmentations to maintain the alignment of images
        and masks. Only used if aug_dict specifies augmentations.

    Returns
    -------
    tuple of (generator, list):
        If val_split is 0, only the training data generator
        and file names are returned.
    tuple of (generator, generator, list, list):
        If val_split > 0, training and validation generators
        are returned along with the filenames used for each
        split. The returned order is training generator,
        validation generator, training file names, validation
        file names.
    """
    # Initialise the datagen arguments
    dg_init_args = {'validation_split': val_split}
    augment = aug_dict is not None and len(aug_dict) > 0
    if augment:
        # add the augmentation arguments if provided
        dg_init_args.update(aug_dict)

    # Instantiate the data generator
    datagen = ImageDataGenerator(**dg_init_args, rescale=rescale)

    # Create the training data image and mask flows
    # Note that the seed is used if augmentations are
    # being used
    train_img_gen = datagen.flow_from_dataframe(img_df,
                                                img_dir,
                                                y_col=None,
                                                class_mode=None,
                                                shuffle=None,
                                                subset='training',
                                                batch_size=batch_size,
                                                seed=aug_seed if augment else None
                                                )

    train_msk_gen = datagen.flow_from_dataframe(msk_df,
                                                msk_dir,
                                                y_col=None,
                                                class_mode=None,
                                                shuffle=None,
                                                subset='training',
                                                color_mode='grayscale',
                                                batch_size=batch_size,
                                                seed=aug_seed if augment else None
                                                )
    # Combine
    train_gen = (pair for pair in zip(train_img_gen, train_msk_gen))

    # If val_split is greater than 0, i.e. the data is actually split
    # create the validation data generators
    val_gen = None
    if val_split > 0:
        if augment:
            # Create new datagenerator for validation data if
            # training data is being augmented
            val_datagen = ImageDataGenerator(validation_split=val_split,
                                             rescale=rescale)
        else:
            # If no augmentations are applied to training data
            # reuse the training datagenerator
            val_datagen = datagen

        val_img_gen = val_datagen.flow_from_dataframe(img_df,
                                                      img_dir,
                                                      y_col=None,
                                                      class_mode=None,
                                                      shuffle=None,
                                                      subset='validation',
                                                      batch_size=batch_size
                                                      )

        val_msk_gen = val_datagen.flow_from_dataframe(msk_df,
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
        # If no split has been done just return train_gen and train filepaths
        return train_gen, train_img_gen.filepaths
    else:
        # Otherwise return train_gen and val_gen and also
        # the file paths that fall into each split

        train_fps = train_img_gen.filepaths
        val_fps = val_img_gen.filepaths

        return train_gen, val_gen, train_fps, val_fps
