import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from miscellaneous import datasets, metrics, utilities
from models import architectures
from variational import operators, solvers

############################### Load informations ################
# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get configuration path.
cfg = utilities.load_toml("BRAINCT")

gt_data = torch.Tensor(np.load("./data/COULE/test/gt.npy"))
# is_data = torch.Tensor(np.load("./data/MAYO/test/mayo_icp_J5_H6.npy"))

######################## Define forward problem parameters
# Get data shape.
m, n = cfg["m"], cfg["n"]
N = len(gt_data)


######################## Define forward problem parameters
# Define the operator A

k_size = cfg["kernel_size"]
sigma = cfg["sigma_blur"]
kernel = utilities.get_gaussian_kernel(k_size, sigma)
A = operators.ConvolutionOperator(kernel, img_shape=(m, n))


# Initialize the solver for post-processing of NN outputs
postprocessor = solvers.ChambollePockTpV(A)

######################## Run the algorithm
TpV_solver = solvers.ChambollePockTpV(A)
K_cp = -1  # [100, 100, 50, 50]  # [-1, -1, -5, -5]


# Initialize the RMSE and SSIM vectors
H = cfg["H"]
RE_vec = np.zeros((N, H + 1))
RMSE_vec = np.zeros((N, H + 1))
SSIM_vec = np.zeros((N, H + 1))
TpV_vec = np.zeros((N, H + 1))

# Compute the reconstruction for each test image.
for i in range(N): 
    print(f"\nImage {i+1} / {N}")
    # Load the actual data for the h-th outer iteration
    x_true = gt_data[i].numpy().flatten()

    # Compute the noisy sinogram
    y = A(x_true)
    y_delta = y + utilities.get_gaussian_noise(y, cfg["noise_level"])

    # Compute FBP reconstruction
    x_init = torch.tensor(y_delta.reshape((1, m, n)), requires_grad=True)

    # Compute reconstruction by iNN or iDeepGuess
    # iterative_predict(preprocessor, weights_path, H, y_delta, K_h=5, model_name="iDG", device="cpu")
    start_time = time.time()
    x_out = utilities.iterative_predict(
        postprocessor,
        weights_path="./model_weights/COULE/incDG_MSELoss_toGT/",  # FbpDG_MSEloss_toGT
        model_name="incDG_toGT_100epoche",
        H=H,
        y_delta=y_delta,
        x_start=x_init,
        device=device,
        K_h=K_cp,  #  if K=0: iNN
    )
    end_time = time.time()

    # Compute the metrics and save them into the corresponding array
    for h in range(H + 1):
        RE_vec[i, h] = metrics.np_RE(x_out[0, h], x_true.reshape((m, n)))
        RMSE_vec[i, h] = metrics.RMSE(x_out[0, h], x_true.reshape((m, n)))
        SSIM_vec[i, h] = metrics.SSIM(x_out[0, h], x_true.reshape((m, n)))
        TpV_vec[i, h] = metrics.TpV(x_out[0, h], p=1)
        print(
            f"\t h: {h} \t RMSE: {RMSE_vec[i, h]:0.4f}, SSIM: {SSIM_vec[i, h]:0.4f}, RE: {RE_vec[i, h]:0.4f}"
        )

        # Save the image into the corresponding folder
        base_path = f"./results/visual/couleTest/incDG/"
        path = base_path + str(h) + "/"
        utilities.create_path_if_not_exists(path)
        plt.imsave(path + str(i) + ".png", x_out[0, h].reshape((m, n)), cmap="gray")

    # Print them out.
    print(
        f"\tTime needed: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}. RMSE: {RMSE_vec[i, -1]:0.4f}, SSIM: {SSIM_vec[i, -1]:0.4f}, RE: {RE_vec[i, -1]:0.4f}"
    )


# Save the metrics
utilities.create_path_if_not_exists("./results/metrics/coule/")
np.save(f"./results/metrics/brainct/RE_incDG.npy", RE_vec)
np.save(f"./results/metrics/brainct/RMSE_incDG.npy", RMSE_vec)
np.save(f"./results/metrics/brainct/SSIM_incDG.npy", SSIM_vec)
np.save(f"./results/metrics/brainct/TpV_incDG.npy", TpV_vec)
