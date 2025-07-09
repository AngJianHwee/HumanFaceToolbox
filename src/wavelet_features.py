import numpy as np
import pywt
from skimage import color

def _apply_threshold(coeffs, threshold_factor):
    """Apply threshold to wavelet coefficients, return filtered coeffs and count of retained."""
    filtered_coeffs = []
    retained_count = 0
    for coeff in coeffs:
        if isinstance(coeff, np.ndarray):
            # Single array (cA)
            max_val = np.max(np.abs(coeff))
            thresh = threshold_factor * max_val
            mask = np.abs(coeff) > thresh
            filtered_coeff = coeff * mask
            retained_count += np.sum(mask)
            filtered_coeffs.append(filtered_coeff)
        else:
            # Tuple of detail coefficients (cH, cV, cD)
            filtered_detail = []
            for detail in coeff:
                max_val = np.max(np.abs(detail))
                thresh = threshold_factor * max_val
                mask = np.abs(detail) > thresh
                filtered_detail.append(detail * mask)
                retained_count += np.sum(mask)
            filtered_coeffs.append(tuple(filtered_detail))
    return filtered_coeffs, retained_count

def extract_wavelet_features(image_np, wavelet='db1', decomposition_level=3, threshold_factor=0.01):
    """
    Performs wavelet decomposition and reconstruction on an image, with optional thresholding.

    Args:
        image_np (numpy.ndarray): The input image as a NumPy array (RGB or grayscale).
        wavelet (str): Name of the wavelet to use (e.g., 'db1', 'haar').
        decomposition_level (int): Level of decomposition.
        threshold_factor (float): Factor to determine threshold for coefficients (e.g., 0.01 for 1% of max).

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The reconstructed image after wavelet transform and thresholding.
            - float: The compression ratio (retained coefficients / total pixels).
            - list: The original wavelet coefficients for each channel (if color) or image (if grayscale).
            - list: The filtered wavelet coefficients for each channel (if color) or image (if grayscale).
    """
    original_coeffs = []
    filtered_coeffs_list = []
    retained_counts = []
    
    if image_np.ndim == 3: # Color image (RGB)
        # Ensure image is float for wavelet transform
        img_float = image_np.astype(float)

        # Ensure dimensions are suitable for wavelet transform
        array_size = min(img_float.shape[:2]) - (min(img_float.shape[:2]) % (2**decomposition_level))
        img_float = img_float[:array_size, :array_size]

        channels = [img_float[:, :, i] for i in range(img_float.shape[2])]
        
        for channel in channels:
            coeffs = pywt.wavedec2(channel, wavelet, level=decomposition_level)
            original_coeffs.append(coeffs)
            filtered_coeffs, retained_count = _apply_threshold(coeffs, threshold_factor)
            filtered_coeffs_list.append(filtered_coeffs)
            retained_counts.append(retained_count)
        
        # Reconstruct each channel
        reconstructed_channels = []
        for filtered_coeff in filtered_coeffs_list:
            rec_channel = pywt.waverec2(filtered_coeff, wavelet)
            reconstructed_channels.append(rec_channel)
        
        reconstructed_image = np.stack(reconstructed_channels, axis=-1)
        reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

        total_pixels = array_size * array_size
        avg_retained_count = sum(retained_counts) / len(retained_counts)
        compression_ratio = avg_retained_count / total_pixels if total_pixels > 0 else 0

    else: # Grayscale image
        img_float = image_np.astype(float)
        
        # Ensure dimensions are suitable for wavelet transform
        array_size = min(img_float.shape[:2]) - (min(img_float.shape[:2]) % (2**decomposition_level))
        img_float = img_float[:array_size, :array_size]

        coeffs = pywt.wavedec2(img_float, wavelet, level=decomposition_level)
        original_coeffs.append(coeffs)
        filtered_coeffs, retained_count = _apply_threshold(coeffs, threshold_factor)
        filtered_coeffs_list.append(filtered_coeffs)
        retained_counts.append(retained_count)

        reconstructed_image = pywt.waverec2(filtered_coeffs_list[0], wavelet)
        reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

        total_pixels = array_size * array_size
        compression_ratio = retained_counts[0] / total_pixels if total_pixels > 0 else 0

    return reconstructed_image, compression_ratio, original_coeffs, filtered_coeffs_list
