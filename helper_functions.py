import matplotlib.pyplot as plt  
import numpy as np               
import torch  


# Helper function to show a single image
def show_single_image(image):
    image_permuted = np.array(image.permute(1, 2, 0))
    plt.imshow(image_permuted, cmap = "gray")
    plt.axis("off")
    plt.show()

# Helper function to show reconstruction results
def show_reconstruction_results(images, reconstructions, cmap = "gray"):
    assert len(images) == len(reconstructions), "Images and reconstructions must have the same length."

    num_images = len(images)

    # If there's only one image, make sure axes is a 2D array
    if num_images == 1:
        fig, axes = plt.subplots(2, 1, figsize=(9, 9))  # Two rows, one column for one image
    else:
        fig, axes = plt.subplots(2, num_images, figsize=(num_images * 3, 6))  # Two rows, num_images columns

    for i in range(num_images):
        # Original images on the top row
        img = images[i].permute(1, 2, 0).cpu().numpy()  # Convert tensor to NumPy
        axes[0, i].imshow(img, cmap=cmap) if num_images > 1 else axes[0].imshow(img, cmap=cmap)
        axes[0, i].axis("off") if num_images > 1 else axes[0].axis("off")
        axes[0, i].set_title("Original") if num_images > 1 else axes[0].set_title("Original")

        # Reconstructions on the bottom row
        recon = reconstructions[i].permute(1, 2, 0).cpu().numpy()
        axes[1, i].imshow(recon, cmap=cmap) if num_images > 1 else axes[1].imshow(recon, cmap=cmap)
        axes[1, i].axis("off") if num_images > 1 else axes[1].axis("off")
        axes[1, i].set_title("Reconstruction") if num_images > 1 else axes[1].set_title("Reconstruction")

    plt.tight_layout()
    plt.show()


# Helper function to plot images on the same row
def show_images_row(images_tensor, titles=None):
    # Ensure the tensor is on the CPU and detach it (if it's part of a computation graph)
    images = images_tensor.cpu().detach()
    
    # Get the number of images
    num_images = images.shape[0]
    
    # Create a figure with 1 row and num_images columns
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    # Handle case where there's only one image (axes is not a list)
    if num_images == 1:
        axes = [axes]

    # Loop through images and axes to display each image
    for i, ax in enumerate(axes):
        # Convert each image to (H, W, C) if it has channels
        img = images[i]
        if img.shape[0] == 1:  # Grayscale image
            img = img.squeeze(0)  # Remove channel dimension
            ax.imshow(img, cmap="gray")
        else:  # RGB image
            img = img.permute(1, 2, 0)  # Convert to (H, W, C)
            ax.imshow(img)

        # Add title if provided
        if titles:
            ax.set_title(titles[i])
        ax.axis("off")  # Hide axes

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

