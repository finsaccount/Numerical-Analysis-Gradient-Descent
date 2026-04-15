import numpy as np
import matplotlib.pyplot as plt
from math import ceil

def visualize_landscape(mse, xi, yi, n=10, clipat=10, arange=[-5,5], brange=[-5,5]):
    a = np.linspace(arange[0], arange[1], n).reshape((1, -1))
    b = np.linspace(brange[0], brange[1], n).reshape((-1, 1))

    z = 0
    batch_size = len(xi)
    for i in range(batch_size):
        z += 1/batch_size * mse(a, b, xi[i], yi[i])
    z = z.clip(max=clipat)

    dex = np.argmin(z)
    row, col = np.unravel_index(dex, z.shape)

    print("Local Minima at: ", a[0, col], b[row, 0])

    print("Approximate value at xi, yi: ", a[0, col] * xi + b[row, 0])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot_surface(a, b, z, alpha=0.5)
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_zlabel("error")

    ax.set_aspect('equal')
    plt.show()

# Goal: Fit a line!

xi = np.array([1,2])
yi = np.array([3,5])

batch_size = len(xi)

# Calculates the mean-square-error function given line weights w and a (xi, yi) pair from the line to approximate
def mse(a_s, b_s, xi, yi):
    return a_s**2 * xi**2 + 2*a_s*(b_s-yi)*xi + yi**2 + b_s**2 - 2*yi*b_s

# Gradient of the mean-square-error function for fitting a line f(x) = ax + b
def grad_mse(a_s, b_s, xi, yi):
    return np.array([
        2*a_s*xi**2 + 2*(b_s-yi)*xi,
        2*a_s*xi + 2*b_s - 2*yi
    ])

# TODO - add shuffle
# def optimize(X, Y, loss_grad, w_start, batch_size=16, lr=0.01, epochs=1):
#     num_batches = ceil(len(X) // batch_size)

#     weight_vector = w_start.copy()
    
#     for e in range(epochs):
#         for b in range(num_batches):
#             # Get batch
#             x = X[b*batch_size:(b+1)*batch_size]
#             y = Y[b*batch_size:(b+1)*batch_size]

#             # Calculate the gradient
#             grad = loss_grad(weight_vector, x, y)
#             grad = lr / len(x) * grad.sum(axis=1)

#             # Update weights
#             weight_vector -= grad

#     return weight_vector

# Line is of the form f(x) = ax + b
a_current = 0.1
b_current = 0.1

lr = 0.1
its = 500

for i in range(its):
    grad = grad_mse(a_current, b_current, xi, yi)

    grad_weighted = lr/batch_size * grad.sum(axis=1)

    a_current = a_current - grad_weighted[0]
    b_current = b_current - grad_weighted[1]

print(a_current, b_current)
print(a_current * xi + b_current)