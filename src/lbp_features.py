import numpy as np
from skimage.feature import local_binary_pattern
from skimage import color

def extract_lbp_features(image_np, radius=3, n_points=None, method='uniform'):
    """
    Extracts LBP (Local Binary Pattern) features from an image.

    Args:
        image_np (numpy.ndarray): The input image as a NumPy array.
        radius (int): Radius of circle (spatial resolution of the operator).
        n_points (int, optional): Number of sampling points on the circle. Defaults to 8 * radius.
        method (str): Method for computing LBP. E.g., 'default', 'ror', 'uniform', 'var'.

    Returns:
        numpy.ndarray: The LBP image.
    """
    if image_np.ndim == 3:
        # Convert to grayscale if it's a color image
        image_gray = color.rgb2gray(image_np)
    else:
        image_gray = image_np
    
    # Convert to uint8 for LBP compatibility if not already
    if image_gray.dtype != np.uint8:
        image_gray = (image_gray * 255).astype(np.uint8)

    if n_points is None:
        n_points = 8 * radius

    lbp_image = local_binary_pattern(image_gray, n_points, radius, method)
    return lbp_image
