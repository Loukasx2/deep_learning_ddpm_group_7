from PIL import Image
import os

def split_grid(image_path, output_dir, grid_size=(28, 28)):
    """
    Splits a large grid image into individual images of size `grid_size`.
    :param image_path: Path to the grid image.
    :param output_dir: Directory to save individual images.
    :param grid_size: Size of individual images (width, height).
    """
    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(image_path)
    img_width, img_height = img.size
    grid_width, grid_height = grid_size

    # Calculate number of rows and columns
    num_cols = img_width // grid_width
    num_rows = img_height // grid_height

    # Extract individual images
    idx = 0
    for row in range(num_rows):
        for col in range(num_cols):
            box = (col * grid_width, row * grid_height,
                   (col + 1) * grid_width, (row + 1) * grid_height)
            cropped_img = img.crop(box)
            cropped_img.save(os.path.join(output_dir, f"img_{idx:05d}.png"))
            idx += 1

    print(f"Saved {idx} images to {output_dir}")

split_grid("/dtu/blackhole/1f/135583/DDPM-Pytorch/default/samples/x0_0.png", "output/sampled_images", grid_size=(28, 28))
