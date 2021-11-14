from tensorflow.keras.layers import (concatenate,
                                     Conv2D,
                                     Conv2DTranspose,
                                     Input,
                                     MaxPooling2D,
                                     BatchNormalization
                                    )
from tensorflow.keras.activations import get as get_activation
from tensorflow.keras.models import Model

def UNet(input_shape=(256, 256, 3), n_filters=64, n_blocks=4, model_name='UNet'):
    """Creates UNet Model
    
    Creates a UNet architecture a la
    Olaf Ronneberger et al.
    
    Parameters
    ----------
    input_shape: tuple of int:
        The dimensions of the input images in the form
        (H,W,C).
    n_filters: int, optional
        The number of filters to use in the first convolution
        block. Default is 64 as used in Ronneberger et al.
    n_blocks: int, optional
        The number of blocks implemented on each path. Default
        is 4 as used in Ronneberger et al.
    model_name: string, optional
        A name to assign to the model.
        
    Returns
    -------
    tensorflow.keras.models.Model:
        The configured U-Net model.
    """
    
    # Reference to input layer
    in_ = Input(input_shape)
    
    # Copy input layer for operations
    x = in_
    
    # Blocks on each side of the UNet architecture
    blocks = n_blocks
    
    # Keep track of layers for concatenation on the DECODE side
    concat_layers = []
    
    ## ---- ENCODER ----
    # An encoder block consists of CONV2D=>CONV2D=>MAXPOOL
    enc_filters = n_filters
    for block in range(blocks):
        
        # Block of Conv2D & MaxPooling2D
        x = Conv2D(filters=enc_filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = Conv2D(filters=enc_filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
        
        # Append to layer list for concatenation
        concat_layers.append(x)
        
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)
        
        # Increase number of filters
        enc_filters *= 2
        
    # Final Conv2D before upsampling
    x = Conv2D(filters=enc_filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters=enc_filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
    
    
    ## ---- DECODER ---- ##
    # A decoder block consists of CONV2DTRANS=>CONCAT=>CONV2D=>CONV2D
    dec_filters = enc_filters / 2
    for block in range(blocks):
        
        # Block of Conv2DTranspose, Concatenate & Conv2D
        x = Conv2DTranspose(dec_filters, kernel_size=(2,2), strides=(2, 2), padding='same') (x)
    
        # Concatenate with corresponding layer output from the encoding side of
        # the UNet.    
        concat = concatenate([concat_layers.pop(), x], axis=3)
        
        x = Conv2D(filters=dec_filters, kernel_size=(3, 3), activation='relu', padding='same')(concat)
        x = Conv2D(filters=dec_filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
    
        dec_filters /= 2
    
    # ---- OUTPUT ----
    # Output a single channel with the probabilty that the pixel
    # is the positive class e.g. foreground, water etc.
    out_ = Conv2D(1, 1, activation='sigmoid')(x)
    
    model_name = f'{model_name}_f{n_filters}_b{n_blocks}'
    model = Model(inputs=in_, outputs=out_, name=model_name)
    
    return model


# Convenience function for adding BN to a Conv2D layer
def conv2D_BN(x, filters, kernel_size=(3, 3), activation='relu',
              padding='same', bn_pos='after'):
    if bn_pos.lower() == 'after':
        x = Conv2D(filters=filters, kernel_size=kernel_size,
                   activation=activation, padding=padding)(x)
        x = BatchNormalization()(x)
    elif bn_pos.lower() == 'before':
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(x)
        x = BatchNormalization()(x)
        x = get_activation(activation)(x)
    else:
        raise ValueError('`bn_pos` must be one of "after", "before".')
    
    return x


# UNet fuction including batchnorm
def UNet_BN(input_shape=(256, 256, 3), n_filters=64, n_blocks=4, bn_pos='after',
            model_name='UNet_BN'):
    """Creates UNet Model
    
    Creates a UNet architecture a la
    Olaf Ronneberger et al.
    
    Parameters
    ----------
    input_shape: tuple of int:
        The dimensions of the input images in the form
        (H,W,C).
    n_filters: int, optional
        The number of filters to use in the first convolution
        block. Default is 64 as used in Ronneberger et al.
    n_blocks: int, optional
        The number of blocks implemented on each path. Default
        is 4 as used in Ronneberger et al.
    bn_pos: str, {"after" | "before"}
        Position of batchnorm layer in the network. Default is
        "after" the layer activation. Alternative is "before"
        the activation.
    model_name: string, optional
        A name to assign to the model.
        
    Returns
    -------
    tensorflow.keras.models.Model:
        The configured U-Net model.
    """
    
    # Reference to input layer
    in_ = Input(input_shape)
    
    # Copy input layer for operations
    x = in_
    
    # Blocks on each side of the UNet architecture
    blocks = n_blocks
    
    # Keep track of layers for concatenation on the DECODE side
    concat_layers = []
    
    ## ---- ENCODER ----
    # An encoder block consists of CONV2D=>CONV2D=>MAXPOOL
    enc_filters = n_filters
    for block in range(blocks):
        
        # Block of Conv2D & MaxPooling2D
        x = Conv2D(filters=enc_filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = conv2D_BN(x, filters=enc_filters, kernel_size=(3, 3), activation='relu', padding='same',
                     bn_pos=bn_pos)
        
        # Append to layer list for concatenation
        concat_layers.append(x)
        
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)
        
        # Increase number of filters
        enc_filters *= 2
        
    # Final Conv2D before upsampling
    x = Conv2D(filters=enc_filters, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = conv2D_BN(x, filters=enc_filters, kernel_size=(3, 3), activation='relu', padding='same',
                     bn_pos=bn_pos)
    
    
    ## ---- DECODER ---- ##
    # A decoder block consists of CONV2DTRANS=>CONCAT=>CONV2D=>CONV2D
    dec_filters = enc_filters / 2
    for block in range(blocks):
        
        # Block of Conv2DTranspose, Concatenate & Conv2D
        x = Conv2DTranspose(dec_filters, kernel_size=(2,2), strides=(2, 2), padding='same') (x)
    
        # Concatenate with corresponding layer output from the encoding side of
        # the UNet.    
        concat = concatenate([concat_layers.pop(), x], axis=3)
        
        x = Conv2D(filters=dec_filters, kernel_size=(3, 3), activation='relu', padding='same')(concat)
        x = conv2D_BN(x, filters=dec_filters, kernel_size=(3, 3), activation='relu', padding='same',
                     bn_pos=bn_pos)
    
        dec_filters /= 2
    
    # ---- OUTPUT ----
    # Output a single channel with the probabilty that the pixel
    # is the positive class e.g. foreground, water etc.
    out_ = Conv2D(1, 1, activation='sigmoid')(x)
    
    model_name = f'{model_name}_f{n_filters}_b{n_blocks}_bn{bn_pos}'
    model = Model(inputs=in_, outputs=out_, name=model_name)
    
    return model