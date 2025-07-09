import numpy as np
from skimage import color, feature, exposure

def extract_hog_features(image_np, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys'):
    """
    Extracts HOG (Histogram of Oriented Gradients) features from an image.

    Args:
        image_np (numpy.ndarray): The input image as a NumPy array (preferably grayscale).
        orientations (int): Number of gradient orientations.
        pixels_per_cell (tuple): Size (in pixels) of a cell.
        cells_per_block (tuple): Number of cells in each block.
        visualize (bool): If True, also return the HOG image for visualization.
        block_norm (str): Normalisation method for blocks.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The HOG feature vector.
            - numpy.ndarray (optional): The HOG image (if visualize is True).
    """
    if image_np.ndim == 3:
        # Convert to grayscale if it's a color image
        image_gray = color.rgb2gray(image_np)
    else:
        image_gray = image_np

    features, hog_image = feature.hog(
        image_gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=visualize,
        block_norm=block_norm
    )

    if visualize:
        # Rescale HOG image for better visualization
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        return features, hog_image_rescaled
    else:
        return features
