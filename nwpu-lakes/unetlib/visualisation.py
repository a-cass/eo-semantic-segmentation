from matplotlib import pyplot as plt
import numpy as np
from pandas.core.frame import DataFrame
from skimage.io import imread


def plot_labels(img_path, label_path, overlay=False, **kwargs):
    """Plot source image and label image.

    Parameters
    ----------
    img_path, label_path : str
        Paths to an image and its corresponding label.
    overlay : bool, default=False
        If true, the mask is plotted on the same axis as the image
        with some transparency applied. Otherwise the image and mask are
        plotted on separate axes.
    **kwargs : dict, optional
        Keyword arguments to be passed to `matplotlib.pyplot.subplots`.
        Refer to the `matplotlib.pyplot.subplots` documentation for
        a list of all possible arguments.
    """
    original = imread(img_path)
    labelled = imread(label_path)
    
    if not overlay:
        fig, axes = plt.subplots(1, 2, **kwargs)
        axes = axes.flatten()
        
        axes[0].imshow(original)
        axes[0].axis('off')
        axes[0].set_title('Original')

        axes[1].imshow(labelled, cmap='gray')
        axes[1].axis('off')
        axes[1].set_title('Labelled')
        
    else:
        fig, ax = plt.subplots(1,1, **kwargs)
        ax.imshow(original)
        ax.imshow(labelled, cmap='bwr_r', alpha=0.4)
        ax.axis('off')


def plot_batch(img_batch, msk_batch=None, n_images=None, n_cols=2,
               figsize=(20, 10)):
    """Plot batch of images

    Parameters
    ----------
    img_batch: numpy.ndarray or tensorflow.Tensor
        Batch of images. Dimensions should be (N, H, W, C).
    msk_batch: numpy.ndarray or tensorflow.Tensor, optional
        Batch of mask images corresponding to each image in
        `img_batch`.
    n_images: int, optional
        Number of images (or image/mask pairs) to plot. By
        default all images are plotted.
    n_cols: int, default=2
        Number of columns in plot. Number of rows will be
        determined from this.
    figsize: tuple of int, default=(20, 10)
        Width and height of figure in inches.
    """
    # Batches must be of dimensions N,H,W,C
    assert img_batch.ndim == 4

    # Set n_images to length of batch if not provided.
    n_images = n_images or len(img_batch)

    if msk_batch is not None:
        assert msk_batch.ndim == 4

        # Batches must contain the same number of images.
        assert len(img_batch) == len(msk_batch)

        # Interleave masks with images
        img_batch = np.insert(img_batch, np.arange(1, len(msk_batch) + 1),
                              msk_batch, axis=0)

        # Double n_images to account for masks
        n_images *= 2

    # Set number of rows
    n_rows = int(np.ceil(n_images / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.ravel()

    for i, ax in zip(range(n_images), axes):
        img = img_batch[i]
        cmap = None
        if img.shape[-1] == 1:
            img = img[:, :, 0]
            cmap = 'gray'

        ax.imshow(img, cmap=cmap)
        ax.axis('off')

    # Turn off any axes that do not have data
    for ax in axes:
        if not ax.has_data():
            ax.axis('off')


def plot_model_history(history, metrics=['loss'], best=['min'], title=None,
                       n_cols=2, figsize=(20, 10)):
    """Plot model training history.
    
    Creates a plot of the results of each metric on the training
    and validation data. "Best" values are highlighted according
    to the methods in `best`.
    
    Parameters
    ----------
    history: pd.DataFrame or dict
        A model's history in either dictionary or DataFrame
        format.
    metrics: list of str, default=['loss']
        List of metrics to plot, one per axis. By default only
        the loss/val_loss will be plotted. For available metrics
        examin the columns of `history`.
    best: list of str, default=['min']
        Method for determining the best metric value. Must align
        with the metrics list i.e. metrics[0] will have its best
        value determined by best[0]. By default just the minimum
        is used for the loss metric. Options are "min" or "max".
    title: str, optional
        Title for the figure. Default setting is to use no title.
    n_cols: int, default=2
        Number of columns to split the metrics over. Number of rows
        will be determined from this.
    figsize: tuple of int, default=(20, 10)
        Width and height of figure in inches.

    Returns
    -------
    matplotlib.figure.Figure
        A figure containing the axes with the metrics.
    """
    if isinstance(history, DataFrame):
        pass
        
    elif isinstance(history, dict):
        history = DataFrame(history)
    
    else:
        raise TypeError('`history` must be dictionary or DataFrame object.')
        
    # Configure axes
    n_rows = int(np.ceil(len(metrics)/n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Record what values are best for each metric
    if best is None or len(best) != len(metrics):
        raise ValueError('`best` must contain a method for each metric'
                         'in `metrics`.')
    best_vals = {m: b for m, b in zip(metrics, best)}
    invalid_best = set(best_vals.values()).difference(set(['min', 'max']))
    if len(invalid_best) > 0:
        raise ValueError('`best` must contain either "min" or "max" '
                         f'but user specified: {",".join(invalid_best)}')
    
    # Plot the data
    for m, ax in zip(metrics, axes.ravel()):
        val_m = f'val_{m}'
        
        # plot metric and its validation counterpart
        history[[m, val_m]].plot(ax=ax)
        
        # Determine best x (epoch) and y (metric) values
        best_x = getattr(history[val_m], f'arg{best_vals[m]}')()
        best_y = getattr(history[val_m], best_vals[m])()
        
        # Plot best value and lines
        ax.plot([best_x] * 2, [0, best_y], color='g')
        ax.plot([0, best_x], [best_y] * 2)
        ax.scatter(best_x, best_y, s=100, marker='o', color='g')
        
        # Set the axis title
        ax.set_title((f'{m.capitalize()}\n{best_vals[m].capitalize()}'
                      f'@ Epoch:{best_x}, Val:{best_y:.4f}'))
        
        # Set the axis limits
        ax.set_ylim(0, 1)
        ax.set_xlim(0, len(history))

    # Set figure title
    fig.suptitle(title)
    
    # Turn off any axes that do not have data
    for ax in axes.ravel():
        if not ax.has_data():
            ax.axis('off')
    
    return fig
