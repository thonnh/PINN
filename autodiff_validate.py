import numpy as np
import matplotlib.pyplot as plt
#%% Numerical differentiation
# Example data
L = 5
x = np.linspace(0, 5, 10)  # Generate more data points for better accuracy
w = (x**4) - (2*L*x**3) + (L**2*x**2)

# Function to calculate the first derivative using central difference
def central_difference(x, y):
    dy_dx = np.zeros_like(y)
    h = x[1] - x[0]  # Assuming evenly spaced x values
    dy_dx[1:-1] = (y[2:] - y[:-2]) / (2*h)  # Central difference for inner points
    dy_dx[0] = (y[1] - y[0]) / h            # Forward difference at the start
    dy_dx[-1] = (y[-1] - y[-2]) / h         # Backward difference at the end
    return dy_dx

# Calculate the derivatives
dy_dx = central_difference(x, w)
d2y_dx2 = central_difference(x, dy_dx)
d3y_dx3 = central_difference(x, d2y_dx2)
d4y_dx4 = central_difference(x, d3y_dx3)

# Print the fourth derivative to check values
print("Fourth Derivative:", d4y_dx4)

# Plot the original data and its derivatives
plt.figure(figsize=(10, 8))
plt.ylim(-70, 70) 
plt.plot(x, w, label='Original Data (w)')
plt.plot(x, dy_dx, label='First Derivative (dw/dx)')
plt.plot(x, d2y_dx2, label='Second Derivative (d²w/dx²)')
plt.plot(x, d3y_dx3, label='Third Derivative (d³w/dx³)')
plt.plot(x, d4y_dx4, label='Fourth Derivative (d⁴w/dx⁴)')

plt.xlabel('x')
plt.ylabel('w and its derivatives')
plt.legend(loc='lower right')  # 'loc' parameter positions the legend at the best location
plt.title('Original Data and Its Derivatives')
plt.grid(True)
plt.show()


#%% auto differentiation 
import torch
import matplotlib.pyplot as plt

x = torch.linspace(0, 5, 10, requires_grad=True)
w = (x**4) - (2*L*x**3) + (L**2*x**2)

# Calculate the first derivative
dwdx = torch.autograd.grad(w, x, torch.ones_like(w), create_graph=True)[0]

# Calculate the second derivative
d2wdx2 = torch.autograd.grad(dwdx, x, torch.ones_like(dwdx), create_graph=True)[0]

# Calculate the third derivative
d3wdx3 = torch.autograd.grad(d2wdx2, x, torch.ones_like(d2wdx2), create_graph=True)[0]

# Calculate the fourth derivative
d4wdx4 = torch.autograd.grad(d3wdx3, x, torch.ones_like(d3wdx3), create_graph=True)[0]

# Convert results to numpy for plotting
x_np = x.detach().numpy()
w_np = w.detach().numpy()
dwdx_np = dwdx.detach().numpy()
d2wdx2_np = d2wdx2.detach().numpy()
d3wdx3_np = d3wdx3.detach().numpy()
d4wdx4_np = d4wdx4.detach().numpy()

# Plot the original data and its derivatives
plt.figure(figsize=(10, 8))
plt.ylim(-70, 70) 
plt.plot(x_np, w_np, label='Original Data (w)')
plt.plot(x_np, dwdx_np, label='First Derivative (dw/dx)')
plt.plot(x_np, d2wdx2_np, label='Second Derivative (d²w/dx²)')
plt.plot(x_np, d3wdx3_np, label='Third Derivative (d³w/dx³)')
plt.plot(x_np, d4wdx4_np, label='Fourth Derivative (d⁴w/dx⁴)')

plt.xlabel('x')
plt.ylabel('w and its derivatives')
plt.legend(loc='lower right')
plt.title('Original Data and Its Derivatives Using PyTorch Autodiff')
plt.grid(True)
plt.show()

# Print the values of the fourth derivative to verify
print("Fourth Derivative:", d4wdx4_np)
#%%
L = 6000
Em = 20000
b = 500
h = 800
q = -25
Ig = (b*h**3)/12
EI = Em*Ig
x = np.linspace(0, 6000, 10)  # Generate more data points for better accuracy
w = ((q*x**4) - (2*q*L*x**3) + (q*L**2*x**2))/(24*EI)
#%% prove that autodiff works on tensor that requires_grad=True but not works when it is detached
import torch
import matplotlib.pyplot as plt

L = 6000
x = torch.linspace(0, 5, 10, requires_grad=True)
y = (x**4) - (2*L*x**3) + (L**2*x**2)

w = y.detach()
print("x:", x)
print("y (original):", y)
print("y_data (detached):", w)

# Calculate the first derivative which will not work with w (y that detach from gradient)
dwdx = torch.autograd.grad(w, x, torch.ones_like(w), create_graph=True)[0]
