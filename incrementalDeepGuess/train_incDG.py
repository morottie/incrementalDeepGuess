import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import miscellaneous.utilities as utilities
from miscellaneous import metrics
from models import architectures
from variational import operators, solvers

import matplotlib.pyplot as plt

############################### Load informations ################
# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get configuration path.
cfg = utilities.load_toml("BRAINCT")
# Set shape parameters
m, n = cfg["m"], cfg["n"]

######################## Define forward problem parameters
# Define the operator A

k_size = cfg["kernel_size"]
sigma = cfg["sigma_blur"]
kernel = utilities.get_gaussian_kernel(k_size, sigma)
A = operators.ConvolutionOperator(kernel, img_shape=(m, n))


# Load training data
gt_data = torch.Tensor(np.load("./data/COULE/train/gt.npy"))
# is_data = torch.Tensor(np.load("./data/coule/train/icp.npy"))

print(gt_data.shape)

# Initialize the solver and the pre-processor
# preprocessor = solvers.ChambollePockTpV(A)

####################### Start training
n_epochs = 100
batch_size = 10

H = cfg["H"]

X_true = gt_data  # .unsqueeze(1)
print(X_true.shape)


X_h = torch.zeros_like(gt_data)
# X_h = X_h.unsqueeze(1)
print(X_h.shape)  


for h in range(H):
    print(h)

    #### Load the actual data for the h-th outer iteration

    # only for INCREMENTAL version to GT
    X_h_plus_1 = X_true

    # only for INCREMENTAL version to IS
    # X_h_plus_1 = is_data[:, h + 1].unsqueeze(1)

    train_data = TensorDataset(
        torch.cat([X_true, X_h], axis=1), X_h_plus_1
    )  
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    ) 

    
    ######################## Define model
    model = architectures.ResUNet(img_ch=1, output_ch=1).to(
        device
    )  

    # Loss function
    loss_fn = nn.MSELoss()  # nn.L1Loss() nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Cycle over the epochs
    print(
        f"Training iDeepGuess {h} -> {h+1} model for {n_epochs} epochs and batch size of {batch_size}."
    )
    loss_total = np.zeros((n_epochs,))
    ssim_total = np.zeros((n_epochs,))

    for epoch in range(n_epochs):
        # Measure the execution time
        epoch_time = time.time()

        # Cycle over the batches
        epoch_loss = 0.0
        ssim_loss = 0.0
        for t, data in enumerate(train_loader):
            X_cat_batch, X_h_plus_1_batch = data

            X_true_batch = X_cat_batch[:, 0].unsqueeze(1)
            X_h_batch = X_cat_batch[:, 1].unsqueeze(1)

            # From x_true, generate x_ris and x_fbp
            X_input_batch = torch.zeros_like(X_true_batch)

            # X_FBP = torch.zeros_like(X_true)
            for i in range(len(X_true_batch)):
                x_true_i = X_true_batch[i].numpy().flatten()
                x_h_i = X_h_batch[i].numpy().flatten()

                # Input to the network
                if h == 0:
                    # Compute the sinogram and add noise, then FBP
                    y_i = A(x_true_i)
                    y_delta_i = y_i + cfg["noise_level"] * np.random.normal(
                        0, 1, y_i.shape
                    )
                    X_input_batch[i] = torch.tensor(
                        y_delta_i.reshape((1, m, n)),
                        requires_grad=True,
                    )
                else:
                    X_input_batch[i] = torch.tensor(
                        x_h_i.reshape((1, m, n)),  
                        requires_grad=True,
                    )

            # Send x and y to gpu
            X_input_batch = X_input_batch.to(device)
            X_h_plus_1_batch = X_h_plus_1_batch.to(device)
            # X_true_batch = X_true_batch.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            X_out_batch = model(X_input_batch)
            loss = loss_fn(
                X_out_batch, X_h_plus_1_batch
            )  # ......  to IS or to GT according to X_h_plus_1
            loss.backward()
            optimizer.step()

            # update loss
            epoch_loss += loss.item()
            ssim_loss += metrics.batch_SSIM(X_out_batch, X_true_batch)

            # print the value of loss
            print(
                f"({t+1}, {epoch+1}) - RMSE: {epoch_loss / (t+1):0.4f} - SSIM: {ssim_loss / (t+1):0.4f}."
                f"Time elapsed: {time.strftime('%H:%M:%S', time.gmtime(time.time() - epoch_time))}.",
                end="\r",
            )
        print("")

        # Update the history
        loss_total[epoch] = epoch_loss / (t + 1)
        ssim_total[epoch] = ssim_loss / (t + 1)

    # Save the weights
    weights_path = f"./model_weights/COULE/incDG_MSELoss_toIS/"

    # Create folder is not exists
    utilities.create_path_if_not_exists(weights_path)

    # Save the weights of the model
    torch.save(
        model.state_dict(), weights_path + f"coule_h_{h}_incDG_toIS_100epoche.pt"
    )

    #### Update for next iteration over h
    if h == 0:
        X_h = torch.Tensor(np.load("./data/COULE/train/y__.npy"))
        
    # Only for incremental training
    X_h_old = X_h.detach().clone()

    with torch.no_grad():
        X_h = X_h.to(device)
        for i in range(0, len(X_true), batch_size):
            X_h[i : i + batch_size] = model(X_h[i : i + batch_size])

        X_h = X_h.cpu()

    fig, axes = plt.subplots(1, 4, figsize=(9, 3))
    titles = ["True", "Target", "Input", "Output"]
    images = [X_true[10], X_h_plus_1[10], X_h_old[10], X_h[10]]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(np.squeeze(img), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    filena = f"z__coule_End_h{h+1}.png"
    plt.savefig(filena, bbox_inches="tight", dpi=300)
    plt.close()
