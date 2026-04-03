import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from miscellaneous import datasets, metrics, utilities
from variational import operators, solvers

from PIL import Image
import glob
import torchvision.transforms as transforms

# Definisci una trasformazione per convertire l'immagine in tensore
transform = transforms.ToTensor()

# Load config
cfg = utilities.load_toml("COULE")

# Set shape parameters
m, n = cfg["m"], cfg["n"]

######################## Define forward problem parameters
# Define the operator A

k_size = cfg["kernel_size"]
sigma = cfg["sigma_blur"]
kernel = utilities.get_gaussian_kernel(k_size, sigma)
A = operators.ConvolutionOperator(kernel, img_shape=(m, n))

# Set the noise std
delta = cfg["noise_level"]



########################### TEST SET .....................................................
print("Processing test set...")

##### Load test set

test_set = torch.Tensor(np.load("./data/COULE/test/gt.npy"))
print(test_set.shape)

tv_data = np.zeros((len(test_set), 1, m, n))
RE = np.zeros((len(test_set), 1))
SSIM = np.zeros((len(test_set), 1))

for idx in range(len(test_set)):
    # Start time
    start_time = time.time()

    # Get x_true from the test set
    x_true = test_set[idx]

    ######################## Get data
    y = A(x_true.flatten())
    y_delta = y + delta * np.random.normal(0, 1, y.shape)

    ######################## Run the algorithm
    TV_solver = solvers.ChambollePockTpV(A)

    epsilon = 1e-4 * np.max(y_delta) * np.sqrt(len(y_delta))
    lmbda = cfg["lmbda"]

    x_tv, err_list = TV_solver(
        y_delta,
        epsilon,
        lmbda=lmbda,
        x_true=x_true.numpy().flatten(),
        maxiter=270, 
        p=0.125,
        starting_point=y_delta.reshape((m * n, 1)),
    )

    # Save
    tv_data[idx, 0] = x_tv.reshape((m, n))

    # Finish
    print(
        f"Test image number {idx+1}/{len(test_set)}\n"
        f"\t y: \t RE = {metrics.np_RE(y_delta.reshape((m, n)) , x_true.numpy().reshape((m, n)) ):0.4f} SSIM = {metrics.SSIM(y_delta.reshape((m, n)), x_true.numpy().reshape((m, n))):0.4f} \n"
        f"\t TpV: \t RE = {metrics.np_RE(tv_data[idx, 0] , x_true.numpy().reshape((m, n)) ):0.4f} SSIM = {metrics.SSIM(tv_data[idx, 0], x_true.numpy().reshape((m, n))):0.4f} \n"
    )
    RE[idx] = metrics.np_RE(tv_data[idx, 0].reshape((m, n)) , x_true.numpy().reshape((m, n)) )
    SSIM[idx] = metrics.SSIM(tv_data[idx, 0], x_true.numpy().reshape((m, n)))


    path_CP = "./results/visual/couleTest/tpv_p0.125/"
    utilities.create_path_if_not_exists(path_CP)
    plt.imsave(
        path_CP + str(idx) + ".png",
        tv_data[idx, 0],
        cmap="gray",
    )
    

# Save in numpy array
np.save("./results/data/coule/tpv_p0.125.npy", tv_data)
np.save(f"./results/metrics/coule/RE_tpv_p0.125.npy", RE)
np.save(f"./results/metrics/coule/SSIM_tpv_p0.125.npy", SSIM)

print(f"SSIM: {np.mean(SSIM):0.4f}, RE: {np.mean(RE):0.4f}")

