import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from miscellaneous import datasets, metrics, utilities
from variational import operators, solvers

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

########################### TEST SET #############################################
print("Processing test set...")

# Load test set
test_set = torch.Tensor(np.load("./data/COULE/test/gt.npy"))

HH = cfg["H"]  # Number of outer iterations
KK = cfg["K"]
processed_data = np.zeros((len(test_set), HH + 1, m * n))

for idx in range(len(test_set)):
    # Start time
    start_time = time.time()

    # Get x_true from the test set
    x_true = test_set[idx]
    print(f"Shape of x: {x_true[0].shape}.")

    ######################## Get data
    y = A(x_true.flatten())
    y_delta = y + delta * np.random.normal(0, 1, y.shape)
    YY = torch.Tensor(y_delta)
    print(f"Shape of y: {YY.shape}.")

    RE = metrics.np_RE(YY.reshape((m, n)), x_true)
    SSIM = 0
    print(f"y delta \t  SSIM: {SSIM:0.4f}, RE: {RE:0.4f}")

    x_zeros = np.zeros((m * n,))

    ######################## Run the algorithm
    iCP_TpV = solvers.iChambollePockTpV(A, cfg)
    x_sol, err_list = iCP_TpV(
        y_delta, x0=y_delta, x_true=x_true.numpy().flatten(), H=HH, index=idx
    )

    # Save
    processed_data[idx] = x_sol

    # Finish
    print(
        f"Image number {idx+1}/{len(test_set)} done.\n"
    )

    # Compute the metrics and save them into the corresponding array
    for h in range(HH + 1):
        # Save the image into the corresponding folder
        base_path = f"./results/visual/incTpV/"
        path = base_path + str(h) + "/"
        utilities.create_path_if_not_exists(path)
        plt.imsave(path + str(idx) + ".png", x_sol[h].reshape((m, n)), cmap="gray")


# Save in numpy array
filenam = f"./results/data/incTpV.npy"
#  np.save(filenam, processed_data)

filenam = f"./results/metrics/coule/RE_incTpV.npy"
np.save(filenam, err_list)
