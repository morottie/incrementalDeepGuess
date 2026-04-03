import numpy as np
import torch
from skimage.metrics import structural_similarity
from miscellaneous import utilities
import matplotlib.pyplot as plt

from . import operators
from miscellaneous import metrics


class ChambollePockTpV:
    def __init__(self, A):
        self.A = A

        self.m, self.n = A.shape

        # Generate Gradient operators
        self.grad = operators.myGradient(
            1, (int(np.sqrt(self.n)), int(np.sqrt(self.n)))
        )

        self.m, self.n = A.shape

    def __call__(
        self,
        b,
        epsilon,
        lmbda,
        x_true=None,
        starting_point=None,
        eta=2e-3,
        maxiter=100,
        p=1,
    ):
        # Compute the approximation to || A ||_2
        nu = np.sqrt(
            self.power_method(self.A, num_iterations=10)
            / self.power_method(self.grad, num_iterations=10)
        )

        # Generate concatenate operator
        K = operators.ConcatenateOperator(self.A, self.grad)

        Gamma = np.sqrt(self.power_method(K, num_iterations=10))

        # Compute the parameters given Gamma
        tau = 1 / Gamma
        sigma = 1 / Gamma
        theta = 1

        # Iteration counter
        k = 0
        J = 5  

        # Initialization
        if starting_point is None:
            x = np.zeros((self.n, 1))
        else:
            x = starting_point

        y = np.zeros((self.m, 1))
        w = np.zeros((2 * self.n, 1))

        xx = x

        # Initialize errors
        rel_err = np.zeros((maxiter, 1))
        residues = np.zeros((maxiter + 1, 1))

        # Stopping conditions
        con = True
        while con and (k < maxiter):
            # Update y
            yy = y + sigma * np.expand_dims(self.A(xx) - b, -1)
            y = max(np.linalg.norm(yy) - (sigma * epsilon), 0) * yy / np.linalg.norm(yy)

            # Compute the magnitude of the gradient
            grad_x = self.grad(xx)
            grad_mag = np.square(grad_x[: len(grad_x) // 2]) + np.square(
                grad_x[len(grad_x) // 2 :]
            )

            # Compute the reweighting factor    
            if k % J == 0:
                W = np.expand_dims(
                    np.power(np.sqrt(eta**2 + grad_mag) / eta, p - 1), -1
                )
                WW = np.concatenate((W, W), axis=0)

            # Update w
            x_grad = np.expand_dims(self.grad(xx), -1)
            ww = w + sigma * x_grad

            abs_ww = np.zeros((self.n, 1))
            abs_ww = np.square(ww[: self.n]) + np.square(ww[self.n :])
            abs_ww = np.concatenate((abs_ww, abs_ww), axis=0)

            lmbda_vec_over_nu = lmbda * WW / nu
            w = lmbda_vec_over_nu * ww / np.maximum(lmbda_vec_over_nu, abs_ww)

            # Save the value of x
            xtmp = x

            # Update x
            x = xtmp - tau * (
                np.expand_dims(self.A.T(y), -1)
                + nu * np.expand_dims(self.grad.T(w), -1)
            )

            # Project x to (x>0)
            x[x < 0] = 0

            # Compute signed x
            xx = x + theta * (x - xtmp)

            # Compute relative error
            if x_true is not None:
                rel_err[k] = np.linalg.norm(
                    xx.flatten() - x_true.flatten()
                ) / np.linalg.norm(x_true.flatten())
                #ssim_list[k] = structural_similarity(xx, x_true, data_range=1)
                #    xx.reshape((self.M, self.N)),
                #    x_true.reshape((self.M, self.N)),
                #    data_range=1,) 
                # metrics.SSIM(xx, x_true)
                
            # Compute the magnitude of the gradient of the actual iterate
            grad_x = self.grad(xx)
            grad_mag = np.expand_dims(
                np.sqrt(
                    np.square(grad_x[: len(grad_x) // 2])
                    + np.square(grad_x[len(grad_x) // 2 :])
                ),
                -1,
            )

            # Compute the value of TpV by reweighting
            ftpv = np.sum(np.abs(W * grad_mag))
            res = np.linalg.norm(self.A(xx) - b, 2) ** 2
            residues[k] = 0.5 * res + lmbda * ftpv

            # Stopping criteria
            c = np.sqrt(res) / (np.max(b) * np.sqrt(self.m))
            d_abs = np.linalg.norm(x.flatten() - xtmp.flatten(), 2)

            if (c >= 9e-6) and (c <= 1.1e-5):
                con = False

            if d_abs < 1e-4 * (1 + np.linalg.norm(xtmp.flatten(), 2)):
                con = False

            # Update k
            k = k + 1
        return x, rel_err

    def power_method(self, A, num_iterations: int):
        b_k = np.random.rand(A.shape[1])

        for _ in range(num_iterations):
            # calculate the matrix-by-vector product Ab
            b_k1 = A.T(A(b_k))

            # calculate the norm
            b_k1_norm = np.linalg.norm(b_k1)

            # re normalize the vector
            b_k = b_k1 / b_k1_norm

        return b_k1_norm

    def compute_obj_value(self, x, y, lmbda, p, eta):
        # Compute the value of the objective function TpV by reweighting
        grad_x = np.expand_dims(self.grad(x), -1)
        grad_mag = np.square(grad_x[: len(grad_x) // 2]) + np.square(
            grad_x[len(grad_x) // 2 :]
        )
        W = np.power(np.sqrt(eta**2 + grad_mag) / eta, p - 1)

        ftpv = np.sum(np.abs(W * np.sqrt(grad_mag)))
        return 0.5 * np.linalg.norm(self.A(x) - y, 2) ** 2 + lmbda * ftpv


class iChambollePockTpV:
    def __init__(self, A, cfg) -> None:
        self.A = A
        self.m, self.n = A.shape

        self.cfg = cfg

        # Shape of the image
        self.M, self.N = int(np.sqrt(self.n)), int(np.sqrt(self.n))

        # Solver
        self.CP_TV = ChambollePockTpV(self.A)

        # Generate Gradient operators
        self.grad = operators.myGradient(1, (self.M, self.N))

    def __call__(self, y, x0=None, x_true=None, H=10, index=1):
        #   print(f"\niCP on image {index}")

        # Flattening
        y = y.flatten()

        # Initialization
        x_vec = np.zeros((H + 1, self.n))
        if x0 is None:
            x_vec[0] = np.zeros((self.n,))
        else:
            x_vec[0] = x0.flatten()

        # Get starting parameters
        epsilon = self.cfg["epsilon_scale"] * np.max(y) * np.sqrt(len(y))
        lmbda = self.cfg["lmbda"]
        p = self.cfg["p"]
        K = self.cfg["K"][0]
        alpha = self.cfg["alpha"]
        alpha_epsilon = self.cfg["alpha_epsilon"]
        eta = self.cfg["eta"]

        err_h = np.zeros((K + 1,)) 
        err_list = [] 
        obj_vec = np.zeros((H + 1,))
        obj_vec[0] = self.compute_obj_value(
            np.expand_dims(x_vec[0], -1), y, lmbda, p, eta
        )
        for h in range(H):
            # Inner step
            x_h, err_h = self.CP_TV(
                y,
                epsilon=epsilon,
                lmbda=lmbda,
                maxiter=K,
                x_true=x_true,
                starting_point=np.expand_dims(x_vec[h], -1),
                p=p,
                eta=eta,
            )
            err_list.append(err_h)
            # Compute the value of the objective function TpV by reweighting
            obj_vec[h + 1] = self.compute_obj_value(x_h, y, lmbda, p, eta)

            if x_true is not None:
                # Print relative error after the h-th iteration
                rel_err = np.linalg.norm(
                    x_h.flatten() - x_true.flatten(), 2
                ) / np.linalg.norm(x_true.flatten(), 2)
                ssim = structural_similarity( 
                    x_h.reshape((self.M, self.N)),
                    x_true.reshape((self.M, self.N)),
                    data_range=1,
                ) 
            # Save the result
            x_vec[h + 1] = x_h[:, 0]
            print(f"{p=:0.3f}, {lmbda=:0.3f}, {K=}, {rel_err=:0.4f}, {ssim=:0.4f}.")

            # Update parameters
            epsilon = epsilon * alpha_epsilon
            p = p * alpha
            if h == 0:
                lmbda = lmbda / 2
            else:
                lmbda = lmbda * (obj_vec[h + 1] / obj_vec[h])

            if h < H - 1:
                K = self.cfg["K"][h + 1]
                err_h = np.zeros((K,))
        err_all = np.concatenate(err_list)
        print(f"err all: {err_all.shape}")

        return x_vec, err_all

    def compute_obj_value(self, x, y, lmbda, p, eta):
        # Compute the value of the objective function TpV by reweighting
        grad_x = np.expand_dims(self.grad(x), -1)
        grad_mag = np.square(grad_x[: len(grad_x) // 2]) + np.square(
            grad_x[len(grad_x) // 2 :]
        )
        W = np.power(np.sqrt(eta**2 + grad_mag) / eta, p - 1)

        ftpv = np.sum(np.abs(W * np.sqrt(grad_mag)))
        return 0.5 * np.linalg.norm(self.A(x) - y, 2) ** 2 + lmbda * ftpv
