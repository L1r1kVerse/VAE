import matplotlib.pyplot as plt  
import numpy as np               
import torch  

# Helper function to show a single image
def show_single_image(image):
    """
    Displays a single image by permuting its dimensions 
    (from channel-first to channel-last) and plotting it 
    with a grayscale colormap.
    """
    image_permuted = np.array(image.permute(1, 2, 0))
    plt.imshow(image_permuted, cmap = "gray")
    plt.axis("off")
    plt.show()

# Helper function to show reconstruction results
def show_reconstruction_results(images, reconstructions, cmap = "gray"):
    """
    Displays original images and their reconstructions. 
    Supports multiple images, arranging them in two rows: 
    the top row for originals and the bottom row for reconstructions. 
    """
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
        axes[0, i].imshow(img, cmap = cmap) if num_images > 1 else axes[0].imshow(img, cmap = cmap)
        axes[0, i].axis("off") if num_images > 1 else axes[0].axis("off")
        axes[0, i].set_title("Original") if num_images > 1 else axes[0].set_title("Original")

        # Reconstructions on the bottom row
        recon = reconstructions[i].permute(1, 2, 0).cpu().numpy()
        axes[1, i].imshow(recon, cmap = cmap) if num_images > 1 else axes[1].imshow(recon, cmap = cmap)
        axes[1, i].axis("off") if num_images > 1 else axes[1].axis("off")
        axes[1, i].set_title("Reconstruction") if num_images > 1 else axes[1].set_title("Reconstruction")

    plt.tight_layout()
    plt.show()


# Helper function to plot images on the same row
def show_images_row(images_tensor, titles = None):
    """
    Displays multiple images in a single row. Supports grayscale and RGB images, 
    with optional titles for each image. Automatically adjusts layout, 
    hides axes, and ensures compatibility for single or multiple images.
    """
    # Ensure the tensor is on the CPU and detach it (if it's part of a computation graph)
    images = images_tensor.cpu().detach()
    
    # Get the number of images
    num_images = images.shape[0]
    
    # Create a figure with 1 row and num_images columns
    fig, axes = plt.subplots(1, num_images, figsize = (15, 5))
    
    # Handle case where there's only one image (axes is not a list)
    if num_images == 1:
        axes = [axes]

    # Loop through images and axes to display each image
    for i, ax in enumerate(axes):
        # Convert each image to (H, W, C) if it has channels
        img = images[i]
        if img.shape[0] == 1:  # Grayscale image
            img = img.squeeze(0)  # Remove channel dimension
            ax.imshow(img, cmap = "gray")
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

# Helper function to visualize the latent space when it is two-dimensional
def plot_latent_space(model, latent_scale = 1.0, grid_size = 25, image_size = 28, figure_size = 15):
    """
    Visualizes the latent space of a 2D latent variable model by generating images 
    corresponding to points in the latent space.
    """
    # Initialize blank canvas for the visualization
    canvas_size = image_size * grid_size
    canvas = np.zeros((canvas_size, canvas_size))

    # Create a grid of latent vector values
    latent_x_values = np.linspace(-latent_scale, latent_scale, grid_size)
    latent_y_values = np.linspace(-latent_scale, latent_scale, grid_size)[::-1]

    # Decode and place each latent vector's generation into the canvas
    for row_index, latent_y in enumerate(latent_y_values):
        for col_index, latent_x in enumerate(latent_x_values):
            # Create a latent vector with the current grid point
            latent_vector = torch.tensor([[latent_x, latent_y]], dtype = torch.float)
            # Decode the latent vector into an image
            decoded_image = model.sample_with_given_latents(latent_vector)
            # Reshape and place the image into the appropriate location on the canvas
            decoded_image = decoded_image[0].detach().cpu().reshape(image_size, image_size)
            row_start = row_index * image_size
            row_end = (row_index + 1) * image_size
            col_start = col_index * image_size
            col_end = (col_index + 1) * image_size
            canvas[row_start:row_end, col_start:col_end] = decoded_image

    # Plot the visualization
    plt.figure(figsize = (figure_size, figure_size))
    plt.title("VAE Latent Space Visualization")
    plt.imshow(canvas, cmap = "Greys_r")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")

    # Configure tick marks for the grid
    start_range = image_size // 2
    end_range = grid_size * image_size + start_range
    tick_positions = np.arange(start_range, end_range, image_size)
    plt.xticks(tick_positions, np.round(latent_x_values, 1))
    plt.yticks(tick_positions, np.round(latent_y_values, 1))

    plt.show()


