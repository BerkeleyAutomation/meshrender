import numpy as np
from PIL import Image

def format_color_vector(value, length):
    retval = value
    if isinstance(value, int):
        value = value / 255.0
    if isinstance(value, float):
        value = np.repeat(value, length)
    if isinstance(value, list):
        value = np.array(value)
    elif isinstance(value, np.ndarray):
        value = value.squeeze()
        if np.issubdtype(value.dtype, np.integer):
            value = (value / 255.0).astype(np.float32)
        if value.ndim != 1:
            raise ValueError('Format vector takes only 1-D vectors')
        if length > value.shape[0]:
            retval = np.hstack((value, np.ones(length - value.shape[0])))
        elif length < value.shape[0]:
            retval = value[:length]
    else:
        raise ValueError('Invalid vector data type')

    return value.squeeze().astype(np.float32)

def format_color_array(value, n_channels):
    if np.issubdtype(value.dtype, np.integer):
        value = (value / 255.0).astype(np.float32)
    if value.shape[1] < n_channels:
        value = np.concatenate((value,
                                np.ones((value.shape[0], n_channels - value.shape[1]))), axis=1)
    value = value[:,:n_channels].astype(np.float32)
    return value

def format_texture_source(texture, target_channels='RGB'):
    """Format a texture as a float32 np array.
    """

    # Pass through None
    if texture is None:
        return None

    # Convert PIL images into numpy arrays
    if isinstance(texture, Image.Image):
        texture = np.array(texture)

    # Format numpy arrays
    if isinstance(texture, np.ndarray):
        if np.issubdtype(texture.dtype, np.integer):
            texture = (texture / 255.0).astype(np.float32)
        elif np.issubdtype(texture.dtype, np.floating):
            texture = texture.astype(np.float32)
        else:
            raise TypeError('Invalid type {} for texture'.format(type(texture)))

        # Format array by picking out correct texture channels or padding
        if texture.ndim == 2:
            texture = texture[:,:,np.newaxis]

        if target_channels == 'R':
            texture = texture[:,:,0]
            texture = texture.squeeze()
        elif target_channels == 'RG':
            if texture.shape[2] == 1:
                texture = np.repeat(texture, 2, axis=2)
            else:
                texture = texture[:,:,(0,1)]
        elif target_channels == 'GB':
            if texture.shape[2] == 1:
                texture = np.repeat(texture, 2, axis=2)
            elif texture.shape[2] > 2:
                texture = texture[:,:,(1,2)]
        elif target_channels == 'RGB':
            if texture.shape[2] == 1:
                texture = np.repeat(texture, 3, axis=2)
            elif texture.shape[2] == 2:
                raise ValueError('Cannot reformat texture with 2 channels into RGB')
            else:
                texture = texture[:,:,(0,1,2)]
        elif target_channels == 'RGBA':
            if texture.shape[2] == 1:
                texture = np.repeat(texture, 4, axis=2)
                texture[:,:,3] = 1.0
            elif texture.shape[2] == 2:
                raise ValueError('Cannot reformat texture with 2 channels into RGBA')
            elif texture.shape[2] == 3:
                texture = np.concatenate((texture,
                                            np.ones((texture.shape[0],
                                                    texture.shape[1],
                                                    1))), axis=2)
        else:
            raise ValueError('Invalid texture channel specification: {}'.format(target_channels))
    else:
        raise TypeError('Invalid type {} for texture'.format(type(texture)))

    return texture.astype(np.float32)
