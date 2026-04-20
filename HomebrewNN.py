import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sig_ab(a, b, xi):
    return sigmoid(a*xi + b)

def sig_dif_a(a, b, xi):
    return xi * sig_dif_b(a, b, xi)

def sig_dif_b(a, b, xi):
    return np.exp(-a*xi - b) / ((1 + np.exp(-a*xi - b))**2)

# w = [a0, b0, c0, a1, b1, c1, a2, b2, c2, c3]
#       0   1   2   3   4   5   6   7   8   9
def M(xi, w):
    s = 0

    # Evaluate the three hidden nodes
    for i in range(3):
        s += w[i+2] * sig_ab(w[i], w[i+1], xi)
    
    # Add final bias term
    s += w[9]

    return s

# gradient
# w = [a0, b0, c0, a1, b1, c1, a2, b2, c2, c3]
#       0   1   2   3   4   5   6   7   8   9
def grad_mse(xi, yi, w):
    grad = np.zeros_like(w)

    for i in range(3):
        # Partial derivatives with respect to a_i
        grad[i*3] = 2 * (yi - M(xi, w)) * -w[i*3+2] * sig_dif_a(w[i*3], w[i*3+1], xi)

        # Partial derivatives with respect to b_i
        grad[i*3+1] = 2 * (yi - M(xi, w)) * -w[i*3+2] * sig_dif_b(w[i*3], w[i*3+1], xi)
        
        # Partial derivatives with respect to c_i
        grad[i*3+2] = 2 * (yi - M(xi, w)) * -1 * sig_ab(w[i*3], w[i*3+1], xi)

    # Final partial - with respect to c_3
    grad[9] = 2 * (yi - M(xi, w)) * -1

    return grad

# np.random.seed(40)
w = np.random.rand(10)

x = np.array([0,1])
y = np.array([0,1])

lr = 0.01
steps = 4000

for s in range(steps):
    grad_step = 0
    loss = 0

    # Calculate gradient & loss
    for i in range(len(x)):
        grad_step += grad_mse(x[i], y[i], w)
        loss += (y[i] - M(x[i], w))**2
    grad_step /= len(x)
    loss /= len(x)

    # Update weights
    w -= lr*grad_step

    if s % 100 == 0:
        print(loss)

y_hat = M(x, w)
print(y_hat)

x_grid = np.linspace(-1,5)
y_hat_grid = M(x_grid, w)

plt.plot(x_grid, y_hat_grid, "g--", x, y, "ro")
plt.show()