import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils import download_image, plot_images, save_plot_as_image
from canny_edge import apply_canny_edge

def main():
    image_url = "https://craftofcoding.wordpress.com/wp-content/uploads/2017/02/lena.jpg"
    
    print("Downloading image for Canny Edge Detection example...")
    original_image = download_image(image_url, grayscale=False)
    gray_image = download_image(image_url, grayscale=True)
    print("Image downloaded.")

    print("Applying Canny Edge Detection...")
    edge_image = apply_canny_edge(gray_image, t_lower=50, t_upper=100)
    print("Canny Edge Detection applied.")

    plot_images(
        [original_image, edge_image],
        ["Original Image", "Canny Edge Detection"],
        figsize=(12, 6)
    )

    save_plot_as_image("canny_edge_detection_example.png")

if __name__ == "__main__":
    main()
