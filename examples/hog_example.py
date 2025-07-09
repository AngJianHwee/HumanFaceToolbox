import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils import download_image, plot_images, save_plot_as_image
# Import the HOG feature extraction function
from hog_features import extract_hog_features

def main():
    image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTJHcWBQijdck5GD1-y-nSOFIEUvbrH7SP7Dg&s"
    
    print("Downloading image for HOG Feature Extraction example...")
    original_image_color = download_image(image_url, grayscale=False)
    original_image_gray = download_image(image_url, grayscale=True)
    print("Image downloaded.")

    print("Extracting HOG features...")
    hog_features, hog_image = extract_hog_features(original_image_color, visualize=True)
    print("HOG features extracted.")
    print(f"Shape of HOG feature vector: {hog_features.shape}")

    # Display HOG image
    plot_images(
        [original_image_color, original_image_gray, hog_image],
        ["Original RGB Image", "Grayscale Image", "Rescaled HOG Features"],
        figsize=(18, 6),
        cmap='gray' # HOG image is grayscale
    )

    # Example of HOG on a cropped region
    print("\nDemonstrating HOG on a cropped region...")
    # Define cropping window (adjust these values based on the image content)
    # These are example coordinates, might need adjustment for different images
    start_x, start_y = 100, 100
    end_x, end_y = start_x + 50, start_y + 70

    cropped_image_color = original_image_color[start_y:end_y, start_x:end_x]
    cropped_image_gray = original_image_gray[start_y:end_y, start_x:end_x]

    cropped_hog_features, cropped_hog_image = extract_hog_features(cropped_image_color, visualize=True)
    print("HOG features extracted from cropped region.")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Full image with rectangle overlay
    rect = plt.Rectangle((start_x, start_y), end_x - start_x, end_y - start_y,
                         edgecolor='red', facecolor='none', linewidth=2)
    ax1.imshow(original_image_color)
    ax1.add_patch(rect)
    ax1.set_title("Original RGB Image\n(with ROI highlighted)")
    ax1.axis('off')

    # Cropped image
    ax2.imshow(cropped_image_color)
    ax2.set_title(f"Cropped Region\n[{start_x}:{end_x}, {start_y}:{end_y}]")
    ax2.axis('off')

    # HOG of cropped region
    ax3.imshow(cropped_hog_image, cmap='hot')
    ax3.set_title("HOG of Cropped Region")
    ax3.axis('off')

    plt.tight_layout()
    plt.show()

    # Save the plot as an image file
    save_plot_as_image("hog_feature_extraction_example.png")


if __name__ == "__main__":
    main()
