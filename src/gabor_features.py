import numpy as np
from skimage.filters import gabor
from joblib import Parallel, delayed
from tqdm import tqdm

def extract_gabor_features(image_np, thetas=None, freqs=None):
    """
    Extracts Gabor features from an image.

    Args:
        image_np (numpy.ndarray): The input image as a NumPy array (preferably grayscale).
        thetas (list, optional): List of orientations (in radians). Defaults to 8 orientations from 0 to pi.
        freqs (list, optional): List of frequencies. Defaults to [0.1, 0.4, 0.7].

    Returns:
        tuple: A tuple containing:
            - list: A list of individual Gabor magnitude responses.
            - numpy.ndarray: The fused feature map (average pooling).
    """
    # Define filter parameters
    if thetas is None:
        thetas = np.arange(0, np.pi, np.pi / 8)  # 8 orientations
    if freqs is None:
        freqs = [0.1, 0.4, 0.7]  # 3 frequencies
    
    total_filters = len(thetas) * len(freqs)

    all_params = [(theta, freq) for theta in thetas for freq in freqs]

    def apply_gabor_filter(args, image):
        theta, freq = args
        filt_real, filt_imag = gabor(image, frequency=freq, theta=theta)
        result = np.sqrt(filt_real**2 + filt_imag**2)
        return result

    magnitude_responses = Parallel(n_jobs=-1)(
        delayed(apply_gabor_filter)((theta, freq), image_np)
        for theta, freq in tqdm(all_params, desc="Processing Gabor Filters", total=total_filters, ncols=100)
    )

    # Average fusion of responses
    stacked = np.stack(magnitude_responses, axis=-1)
    avg_fused = np.mean(stacked, axis=-1)

    return magnitude_responses, avg_fused
