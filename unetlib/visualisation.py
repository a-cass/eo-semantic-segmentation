from matplotlib import pyplot as plt
from numpy import ceil
from skimage.io import imread

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