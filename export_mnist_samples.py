import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


MNIST_PATH = r"C:\Users\hoang\.keras\datasets\mnist.npz"
OUTPUT_DIR = r"D:\TinyML_ESP32\mnist_samples"
GRID_ROWS = 5
GRID_COLS = 5


def export_split(images, labels, split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)

    for i, (image, label) in enumerate(zip(images, labels)):
        out_path = os.path.join(split_dir, f"{split_name}_{i:05d}_label_{label}.png")
        Image.fromarray(image).save(out_path)

    return split_dir


def main():
    data = np.load(MNIST_PATH)
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_dir = export_split(x_train, y_train, "train")
    test_dir = export_split(x_test, y_test, "test")

    plt.figure(figsize=(8, 8))
    total_grid_images = GRID_ROWS * GRID_COLS

    for i in range(total_grid_images):
        plt.subplot(GRID_ROWS, GRID_COLS, i + 1)
        plt.imshow(x_train[i], cmap="gray")
        plt.title(str(y_train[i]))
        plt.axis("off")

    plt.tight_layout()
    grid_path = os.path.join(OUTPUT_DIR, "mnist_grid.png")
    plt.savefig(grid_path, dpi=200)
    plt.close()

    print(f"Saved {len(x_train)} training images to: {train_dir}")
    print(f"Saved {len(x_test)} test images to: {test_dir}")
    print(f"Saved grid image to: {grid_path}")


if __name__ == "__main__":
    main()
