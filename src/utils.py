import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

def download_image(url, grayscale=True):
    """
    Downloads an image from a given URL and returns it as a NumPy array.

    Args:
        url (str): The URL of the image.
        grayscale (bool): If True, converts the image to grayscale. Defaults to True.

    Returns:
        numpy.ndarray: The image as a NumPy array.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

    image_bytes = BytesIO(response.content)
    if grayscale:
        pil_img = Image.open(image_bytes).convert('L') # Convert to grayscale
    else:
        pil_img = Image.open(image_bytes).convert('RGB') # Convert to RGB
    
    return np.array(pil_img)

def plot_images(images, titles, figsize=(12, 6), cmap='gray'):
    """
    Plots a list of images using matplotlib.

    Args:
        images (list): A list of NumPy arrays representing the images.
        titles (list): A list of strings for the titles of each image.
        figsize (tuple): Figure size (width, height) in inches.
        cmap (str): Colormap for displaying images. Defaults to 'gray'.
    """
    num_images = len(images)
    if num_images == 0:
        print("No images to plot.")
        return

    # Determine grid size for plotting
    rows = int(np.ceil(num_images / 2)) if num_images > 1 else 1
    cols = 2 if num_images > 1 else 1
    if num_images == 1:
        rows = 1
        cols = 1

    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.title(title)
        if img.ndim == 2: # Grayscale image
            plt.imshow(img, cmap=cmap)
        else: # Color image (assuming RGB)
            plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def save_plot_as_image(filename):
    """
    Saves the current matplotlib plot as an image file.

    Args:
        filename (str): The name of the file to save the plot as.
    """
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    print(f"Plot saved as {filename}")