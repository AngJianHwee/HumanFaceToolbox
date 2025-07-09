import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils import download_image, save_plot_as_image
from wavelet_features import extract_wavelet_features

def main():
    image_url = "https://i.sstatic.net/3T6Gc.jpg"

    
    print("Downloading image for Wavelet Feature Extraction example...")
    original_image_color = download_image(image_url, grayscale=False)
    print("Image downloaded.")

    print("Extracting Wavelet features with default parameters (db1, level 3, 1% threshold)...")
    reconstructed_image, compression_ratio, original_coeffs, filtered_coeffs_list = \
        extract_wavelet_features(original_image_color)
    print(f"Wavelet features extracted. Compression Ratio: {compression_ratio:.4f}")

    # Function to normalize coefficients for visualization
    def normalize_coeff(coeff):
        if isinstance(coeff, np.ndarray):
            return np.log(np.abs(coeff) + 1)
        return [np.log(np.abs(c) + 1) for c in coeff]

    # Visualize results
    plt.figure(figsize=(15, 15))

    # Original RGB Image
    plt.subplot(5, 5, 1)
    plt.title("Original RGB Image")
    plt.imshow(original_image_color)
    plt.axis('off')

    # Reconstructed RGB Image
    plt.subplot(5, 5, 2)
    plt.title(f"Reconstructed RGB Image\nCompression: {compression_ratio:.4f}")
    plt.imshow(reconstructed_image)
    plt.axis('off')

    # Display wavelet coefficients for each channel and level (for the first channel, e.g., Red)
    if original_coeffs:
        # Assuming original_coeffs[0] is for the first channel (Red)
        coeffs = original_coeffs[0] 
        
        cA = normalize_coeff(coeffs[0])  # Approximation at highest level
        # Details for each level, from highest to lowest decomposition level
        details_levels = [normalize_coeff(d) for d in coeffs[1:]] 

        # Plotting logic for coefficients (example for first channel)
        # This part is simplified; a full plot of all 3 channels and all levels would be very large
        # We'll show cA and the first detail coefficients for each level
        
        # Approximation coefficient
        plt.subplot(5, 5, 6)
        plt.title("cA (Approx.)")
        plt.imshow(cA, cmap='gray')
        plt.axis('off')

        # Detail coefficients for each level
        for level_idx, (cH, cV, cD) in enumerate(details_levels):
            # Plot horizontal detail
            plt.subplot(5, 5, 7 + level_idx * 3)
            plt.title(f"cH{len(details_levels) - level_idx}")
            plt.imshow(cH, cmap='gray')
            plt.axis('off')

            # Plot vertical detail
            plt.subplot(5, 5, 8 + level_idx * 3)
            plt.title(f"cV{len(details_levels) - level_idx}")
            plt.imshow(cV, cmap='gray')
            plt.axis('off')

            # Plot diagonal detail
            plt.subplot(5, 5, 9 + level_idx * 3)
            plt.title(f"cD{len(details_levels) - level_idx}")
            plt.imshow(cD, cmap='gray')
            plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Save the plot as an image file
    save_plot_as_image("wavelet_feature_extraction_example.png")

if __name__ == "__main__":
    main()
