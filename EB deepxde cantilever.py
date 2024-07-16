import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# Define the geometry of the problem (interval from 0 to 1)
geom = dde.geometry.Interval(0, 1)

# Define second, third, and fourth derivatives needed for the PDE
def ddy(x, y):
    return dde.grad.hessian(y, x)

def dddy(x, y):
    return dde.grad.jacobian(ddy(x, y), x)

# Define the PDE (Euler-Bernoulli beam equation)
def pde(x, y):
    dy_xx = ddy(x, y)
    dy_xxxx = dde.grad.hessian(dy_xx, x)
    return dy_xxxx + 1  # Adjust as necessary to match the exact form of your PDE

# Define boundary conditions
def boundary_l(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0)

def boundary_r(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)

# Analytical solution for comparison
def func(x):
    return -(x ** 4) / 24 + x ** 3 / 6 - x ** 2 / 4

# Define boundary conditions
bc1 = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_l)
bc2 = dde.icbc.NeumannBC(geom, lambda x: 0, boundary_l)
bc3 = dde.icbc.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_r)
bc4 = dde.icbc.OperatorBC(geom, lambda x, y, _: dddy(x, y), boundary_r)

# Create the dataset
data = dde.data.PDE(
    geom,
    pde,
    [bc1, bc2, bc3, bc4],
    num_domain=10,
    num_boundary=2,
    solution=func,
    num_test=100,
)

# Define the neural network
layer_size = [1] + [20] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# Create the model
model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])

# Train the model
losshistory, train_state = model.train(iterations=10000)

# Plotting the results
X = geom.uniform_points(1000, True)  # Generate uniform points in the domain for predictions
y_pred = model.predict(X)  # Predict the deflection using the trained model
y_exact = func(X)  # Compute the exact solution for comparison

# Plot the predicted and exact solutions
plt.figure(figsize=(8, 4))
plt.plot(X, y_exact, label="Exact solution", color="tab:grey", alpha=0.6)
plt.plot(X, y_pred, label="PINN solution", color="tab:green")
plt.xlabel("x")
plt.ylabel("Deflection")
plt.legend()
plt.title("Euler-Bernoulli Beam Deflection")
plt.show()
