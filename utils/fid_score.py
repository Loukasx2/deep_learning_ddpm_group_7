import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_gan as tfgan
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os

# Load TensorFlow Hub MNIST classifier
MNIST_MODULE = "https://tfhub.dev/tensorflow/tfgan/eval/mnist/logits/1"
mnist_classifier_fn = tfhub.load(MNIST_MODULE)


# Helper Functions
def pack_images_to_tensor(path, img_size=None):
    """
    Pack images from a directory into a TensorFlow tensor.
    """
    nb_images = len(list(Path(path).rglob("*.png")))
    print(f"Packing {nb_images} images from {path}")
    images = np.empty((nb_images, 28, 28, 1))
    for idx, f in enumerate(tqdm(Path(path).rglob("*.png"))):
        img = Image.open(f)

        # Ensure image is grayscale
        img = ImageOps.grayscale(img)
        # Resize if needed
        if img_size and img.size[:2] != img_size:
            img = img.resize(size=(img_size[0], img_size[1]), resample=Image.BILINEAR)
        img = np.array(img) / 255.0  # Normalize to [0, 1]
        images[idx] = img[..., None]  # Add channel dimension
    images_tf = tf.convert_to_tensor(images, dtype=tf.float32)
    return images_tf


def load_mnist():
    """
    Load MNIST dataset and return it as a TensorFlow tensor.
    """
    transform = transforms.Compose([transforms.Resize(28), transforms.ToTensor()])
    dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=60000, shuffle=False)
    x, y = next(iter(dataloader))
    x = torch.permute(x, (0, 2, 3, 1))  # Convert to (N, H, W, C)
    return tf.convert_to_tensor(x.numpy())


def compute_activations(tensors, num_batches, classifier_fn):
    """
    Compute activations from the given classifier function for input tensors.
    """
    tensors_list = tf.split(tensors, num_or_size_splits=num_batches)
    stack = tf.stack(tensors_list)
    activation = tf.nest.map_structure(
        tf.stop_gradient,
        tf.map_fn(classifier_fn, stack, parallel_iterations=1, swap_memory=True),
    )
    return tf.concat(tf.unstack(activation), 0)


def compute_mnist_stats(mnist_classifier_fn):
    """
    Compute and return activations for the real MNIST dataset.
    """
    mnist = load_mnist()
    num_batches = 1  # Use all images in a single batch
    activations = compute_activations(mnist, num_batches, mnist_classifier_fn)
    return activations


def save_activations(activations, path):
    """
    Save activations to a file.
    """
    np.save(path, activations.numpy())


def load_activations(path):
    """
    Load activations from a file.
    """
    return tf.convert_to_tensor(np.load(path), dtype=tf.float32)


# Main Workflow
def compute_fid_score(epoch_dir, real_activations_path, mnist_classifier_fn):
    """
    Compute the FID score between real and generated images.
    """
    # Load precomputed real activations
    activations_real = load_activations(real_activations_path)

    # Load generated images and compute activations
    print(f"Loading images from {epoch_dir}")
    epoch_images = pack_images_to_tensor(path=epoch_dir)
    print("Computing activations for generated images...")
    activations_fake = compute_activations(epoch_images, num_batches=1, classifier_fn=mnist_classifier_fn)

    # Compute FID score
    print("Computing FID score...")
    fid = tfgan.eval.frechet_classifier_distance_from_activations(
        activations_real, activations_fake
    )
    print(f"FID Score: {fid.numpy()}")
    return fid.numpy()


# Example Execution
if __name__ == "__main__":
    # 1. Compute and save real MNIST activations (only once)
    real_activations_path = "./data/mnist/activations_real.npy"
    if not os.path.exists(real_activations_path):
        print("Computing real MNIST activations...")
        real_activations = compute_mnist_stats(mnist_classifier_fn)
        save_activations(real_activations, real_activations_path)

    # 2. Set path to generated images
    generated_images_path = "default_fid/samples"  # Directory containing sampled images from DDPM

    # 3. Compute FID score for the generated images
    fid_score = compute_fid_score(
        epoch_dir=generated_images_path,
        real_activations_path=real_activations_path,
        mnist_classifier_fn=mnist_classifier_fn,
    )
