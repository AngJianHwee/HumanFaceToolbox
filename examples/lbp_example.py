import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils import download_image, plot_images, save_plot_as_image
# Import the LBP feature extraction function
from lbp_features import extract_lbp_features

def main():
    image_url = "https://craftofcoding.wordpress.com/wp-content/uploads/2017/02/lena.jpg"
    
    print("Downloading image for LBP Feature Extraction example...")
    original_image_color = download_image(image_url, grayscale=False)
    original_image_gray = download_image(image_url, grayscale=True)
    print("Image downloaded.")

    print("Extracting LBP features with default parameters...")
    lbp_image_default = extract_lbp_features(original_image_color)
    print("LBP features extracted.")

    print("\nExtracting LBP features with custom parameters (e.g., larger radius, different method)...")
    lbp_image_custom = extract_lbp_features(original_image_color, radius=5, method='ror')
    print("Custom LBP features extracted.")

    # Display LBP images
    plot_images(
        [original_image_color, original_image_gray, lbp_image_default, lbp_image_custom],
        ["Original RGB Image", "Grayscale Image", "LBP (Default)", "LBP (Custom: ROR, R=5)"],
        figsize=(15, 8),
        cmap='viridis' # LBP image is typically visualized with a colormap
    )

    # Optional: Show histogram of LBP values for default
    plt.figure(figsize=(8, 4))
    unique_vals, counts = np.unique(lbp_image_default.ravel(), return_counts=True)
    plt.bar(unique_vals, counts / float(counts.sum()), width=0.5, color='k')
    plt.xlabel("Local Binary Pattern Values")
    plt.ylabel("Frequency")
    plt.title("LBP Histogram (Default Parameters)")
    plt.grid(True)
    plt.show()

    # Save the plot as an image file
    save_plot_as_image("lbp_feature_extraction_example.png")

if __name__ == "__main__":
    main()
