from tensorflow.keras.metrics import MeanIoU
from tensorflow import int32
import tensorflow.keras.backend as K


class BinaryMeanIoU(MeanIoU):
    """Computes the mean Intersection-Over-Union metric.
    
    The standard MeanIoU metric assumes the true values and
    predictions are integers corresponding to class numbers.
    However, in cases of binary semantic segmentation where
    networks end with a Sigmoid activation the outputs are
    floating point values between 0 and 1.
    
    This BinaryMeanIoU metric therefore accepts a threshold
    value used to map the Sigmoid activations to the classes
    prior to calculating the Mean IoU.
    
    
    Parameters
    ----------
    threshold: float, optional
        Probability threshold value. Values greater than or equal
        to this will be assigned to the positive class, 1. Default
        is 0.5.
    name: str, optional
         Name of the metric instance.
    dtype: TensorFlow data type, optional
    
    """
    def __init__(self, threshold=0.5, name='binary_mean_iou', dtype=None):
        super(BinaryMeanIoU, self).__init__(num_classes=2, name=name,
                                            dtype=dtype)
        self.threshold = threshold
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # If predictions are not floating point they are just passed through
        if y_pred.dtype.is_floating:
            # Apply the threshold value
            y_pred = K.cast(y_pred > self.threshold, int32)
        # Pass new y_pred to standard MeanIoU function
        super(BinaryMeanIoU, self).update_state(y_true, y_pred, sample_weight)
