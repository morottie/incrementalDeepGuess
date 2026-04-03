import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from models import architectures


def create_path_if_not_exists(path):
    """
    Check if the path exists. If this is not the case, it creates the required folders.
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def viz_and_compare(img, title=None, save_path=None):
    """
    Takes as input a tuple of images and (optionally) titles, and returns
    a sequence of visualization of images with the given title.
    The tuple of images is assumed to have shape (m, n) or (1, m, n).
    If a single image is given, it is visualized.
    """
    if type(img) is not tuple:
        if len(img.shape) == 3:
            img = img[0]

        plt.imshow(img, cmap="gray")
        plt.axis("off")
        if title is not None:
            plt.title(title)
        if save_path is not None:
            plt.savefig(save_path)
        plt.tight_layout()
        plt.show()

    else:
        plt.figure(figsize=(len(img) * 2, 2))
        for i, x in enumerate(img):
            if len(x.shape) == 3:
                x = x[0]

            plt.subplot(1, len(img), i + 1)
            plt.imshow(x, cmap="gray")
            plt.axis("off")
            if title is not None:
                plt.title(title[i])
        if save_path is not None:
            plt.savefig(save_path)
        plt.tight_layout()
        plt.show()


# Noise is added by noise level
def get_gaussian_noise(y, noise_level=0.01):
    noise = np.random.normal(size=y.shape)  # random gaussian distribution
    noise /= np.linalg.norm(noise.flatten(), 2)  # frobenius norm
    return noise * noise_level * np.linalg.norm(y.flatten(), 2)


def load_toml(data_name):
    """Load TOML data from file, given dataset name."""
    from pip._vendor import tomli

    with open(f"./config/{data_name}.toml", "rb") as f:
        tomo_file = tomli.load(f)
        return tomo_file


def normalize(x):
    """Given an array x, returns its normalized version (i.e. the linear projection into [0, 1])."""
    return (x - x.min()) / (x.max() - x.min())


def iterative_predict(
    postprocessor,
    weights_path,
    H,
    y_delta,
    x_start,
    K_h=5,
    model_name="incDG",
    device="cpu",
):
    """Given a preprocessor, a model weight path, a number of iterations H,
    corrupted data y_delta and other parameters, apply the preprocessing + NN scheme."""

    # Get configuration
    cfg = load_toml("BRAINCT")

    # Initialization
    X_h = np.zeros((1, H + 1, cfg["m"], cfg["n"]))
    X_h[0, 0] = x_start.detach().numpy().reshape((cfg["m"], cfg["n"]))

    for h in range(H):
        ######################## Define model and load weights
        model = architectures.ResUNet(img_ch=1, output_ch=1).to(device)
        model.load_state_dict(
            torch.load(os.path.join(weights_path, f"coule_h_{h}_{model_name}.pt"))
        )

        # Send X_start to the GPU
        x_init = X_h[0, h]
        x_init = torch.tensor(
            x_init.reshape((1, 1, cfg["m"], cfg["n"])), requires_grad=True
        )
        x_init = x_init.to(device)

        # Forward + move to cpu
        X_h[0, h + 1] = model(x_init.float()).cpu().detach().numpy()

        K = K_h if type(K_h) is int else K_h[h]
        if K > 0:  # execute postprocessing
            # Set parameters
            epsilon = (
                cfg["epsilon_scale"]
                * np.max(y_delta)
                * np.sqrt(len(y_delta))
                * (cfg["alpha_epsilon"]) ** h
            )
            lmbda = cfg["lmbda"] / (h + 1)
            # print(f"\t\t lambda = {lmbda}")
            p = 1 * cfg["alpha"] ** h

            # Compute and overwrite real h-th solution from NN output
            temp = torch.tensor(  # X_h[0, h + 1]
                postprocessor(
                    y_delta,
                    epsilon,
                    lmbda,
                    maxiter=K,
                    p=p,
                    starting_point=np.expand_dims(X_h[0, h + 1].flatten(), -1),
                ).reshape((1, 1, cfg["m"], cfg["n"])),
                requires_grad=True,
            )
            X_h[0, h + 1] = temp.cpu().detach().numpy()
    return X_h


def get_gaussian_kernel(k, sigma):
    """
    Creates gaussian kernel with kernel size 'k' and a variance of 'sigma'
    """
    ax = np.linspace(-(k - 1) / 2.0, (k - 1) / 2.0, k)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)
