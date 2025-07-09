import cv2
import numpy as np

def apply_canny_edge(image_np, t_lower=50, t_upper=300):
    """
    Applies Canny Edge Detection to an image.

    Args:
        image_np (numpy.ndarray): The input image as a NumPy array (preferably grayscale).
        t_lower (int): Lower threshold for the Canny algorithm.
        t_upper (int): Upper threshold for the Canny algorithm.

    Returns:
        numpy.ndarray: The resulting edge map.
    """
    if image_np.ndim == 3:
        # Convert to grayscale if it's a color image
        gray_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image_np

    edge = cv2.Canny(gray_img, t_lower, t_upper)
    return edge
