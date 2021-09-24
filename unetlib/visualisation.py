from matplotlib import pyplot as plt
from numpy import ceil
from skimage.io import imread
from pandas.core.frame import DataFrame

def plot_labels(img_path, label_path, overlay=False, **plot_kwargs):
    """Plot source image and label image.
    """
    
    original = imread(img_path)
    labelled = imread(label_path)
    
    if not overlay:
        fig, axes = plt.subplots(1,2, **plot_kwargs)
        axes = axes.flatten()
        
        axes[0].imshow(original)
        axes[0].axis('off')
        axes[0].set_title('Original')

        axes[1].imshow(labelled, cmap='gray')
        axes[1].axis('off')
        axes[1].set_title('Labelled')
        
    else:
        fig, ax = plt.subplots(1,1, **plot_kwargs)
        ax.imshow(original)
        ax.imshow(labelled, cmap='bwr_r', alpha=0.4)
        ax.axis('off')
        
def plot_batch(img_batch, msk_batch=None, n_images=None, **plot_kwargs):
    """
    """
    # Batches must be of dimensions N,H,W,C
    assert img_batch.ndim == msk_batch.ndim == 4
    
    # Batches must be same size
    assert img_batch.shape[0] == msk_batch.shape[0]
    
    # Set number of rows to 1 per image/mask pair
    n_rows = n_images or img_batch.shape[0]
    n_cols = 2 # image , mask
    
    fig, axes = plt.subplots(n_rows, 2, **plot_kwargs)
    
    for i, row in zip(range(n_images), axes):
        img = img_batch[i]
        msk = msk_batch[i]
        if msk.shape[-1] == 1:
            msk = msk[:,:,0]
        
        row[0].imshow(img)
        row[0].set_title('Image')
        row[0].axis('off')
        row[1].imshow(msk, cmap='gray')
        row[1].set_title('Mask')
        row[1].axis('off')

def plot_model_history(history, metrics=['loss'], best=['min'], title=None,
                       figsize=(20,10), n_cols=2):
    """Plot model training history.
    
    Creates a plot of the results of each metric on the training
    and validation data. "Best" values are highlighted according
    to the methods in `best`.
    
    Parameters
    ----------
    history: pd.DataFrame or dict
        A model's history in either dictionary or DataFrame
        format.
    metrics: list of str, optional
        List of metrics to plot, one per axis. By default only
        the loss/val_loss will be plotted
    best: list of str, optional
        Method for determining the best metric value. Must align
        with the metrics list i.e. metrics[0] will have its best
        value determined by best[0]. By default just the minimum
        is used for the loss metric. Options are "min" or "max".
    title: str, optional
        Title for the figure. Default setting is to use no title.
    figsize: tuple of int, optional
        Width and height of figure in inches. Default dimensions
        are (20, 10)
    n_cols: int, optional
        Number of columns to split the metrics over. Number of rows
        will be determined from this. Default is 2 columns
        
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
    fig, axes = plt.subplots(n_rows, n_cols,figsize=figsize)
    
    
    # Record what values are best for each metric
    if best is None or len(best) != len(metrics):
        raise ValueError('`best` must contain a method for each metric in `metrics`.')
    best_vals = {m:b for m,b in zip(metrics, best)}
    invalid_best = set(best_vals.values()).difference(set(['min','max']))
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
        ax.scatter(best_x, best_y, s=100, marker='o',color='g')
        
        # Set the axis title
        ax.set_title((f'{m.capitalize()}\n{best_vals[m].capitalize()}'
                 f'@ Epoch:{best_x}, Val:{best_y:.4f}'))
        
        # Set the axis limits
        ax.set_ylim(0,1)
        ax.set_xlim(0,len(history))
        
    
    # Set figure title
    fig.suptitle(title)
    
    # Turn off any axes that do not have data
    for ax in axes.ravel():
        if not ax.has_data():
            ax.axis('off')
    
    return fig