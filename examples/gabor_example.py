import sys
import os
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils import download_image, plot_images, save_plot_as_image
# Import the Gabor feature extraction function
from gabor_features import extract_gabor_features

def main():
    image_url = "https://craftofcoding.wordpress.com/wp-content/uploads/2017/02/lena.jpg"
    
    print("Downloading image for Gabor Feature Extraction example...")
    original_image_gray = download_image(image_url, grayscale=True)
    original_image_color = download_image(image_url, grayscale=False)
    print("Image downloaded.")

    print("Extracting Gabor Features with default parameters...")
    magnitude_responses, avg_fused = extract_gabor_features(original_image_gray)
    print("Gabor Features extracted.")

    # Example of specifying custom parameters
    print("\nExtracting Gabor Features with custom parameters (e.g., fewer orientations, different frequencies)...")
    custom_thetas = np.arange(0, np.pi, np.pi / 4) # 4 orientations
    custom_freqs = [0.2, 0.6] # 2 frequencies
    custom_magnitude_responses, custom_avg_fused = extract_gabor_features(
        original_image_gray, 
        thetas=custom_thetas, 
        freqs=custom_freqs
    )
    print("Custom Gabor Features extracted.")

    # Prepare images and titles for plotting for default parameters
    images_to_plot_default = [original_image_color] + magnitude_responses + [avg_fused]
    titles_to_plot_default = ["Original Image"] + [f"Default Gabor Response {i+1}" for i in range(len(magnitude_responses))] + ["Default Fused Features (Average)"]

    images_to_plot_default = images_to_plot_default[0:6]  # Limit to first 6 images for better visualization
    titles_to_plot_default = titles_to_plot_default[0:6]  # Limit to first 6 titles for better visualization

    # Adjust figsize for a larger number of plots for default parameters
    num_plots_default = len(images_to_plot_default)
    
    rows_default = int(np.ceil(num_plots_default / 6)) # Max 6 columns for better visualization
    cols_default = 6
    figsize_default = (20, rows_default * num_plots_default) # Adjust height based on number of rows

    plot_images(images_to_plot_default, titles_to_plot_default, figsize=figsize_default, cmap='viridis')
    print(f"Shape of default Gabor feature vector: {magnitude_responses[0].shape}")
    print(f"Shape of default fused feature vector: {avg_fused.shape}")
    
    # Save the plot as an image file
    save_plot_as_image("gabor_feature_extraction_example.png")

if __name__ == "__main__":
    main()
