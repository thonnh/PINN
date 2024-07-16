import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#%%
def exact_solution(L, E, I, q, x):
    "Defines the analytical solution to the Euler-Bernoulli beam problem."
    u = ((q*x**4)-(2*q*L*x**3)+(q*L**2*x**2))/(24*E*I)
    return u

class FCN(nn.Module):
    "Defines a standard fully-connected network in PyTorch"

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
#%%    
L = 6000        # mm
E = 20          # GPa
I = 2.133E+10   # mm^4
q = -25E-3      # kN/mm

x_test = torch.linspace(0,6000,300).view(-1,1)
u_exact = exact_solution(L, E, I, q, x_test)

plt.plot(x_test[:,0], u_exact[:,0], label="Exact solution", color="tab:grey", alpha=0.6)
#%%

# define a neural network to train
# TODO: write code here
pinn = FCN(1,1,64,3)

# define boundary points, for the boundary loss
# TODO: write code here
x_bc0 = torch.tensor(0.).view(-1,1).requires_grad_(True)
x_bcL = torch.tensor(6000.).view(-1,1).requires_grad_(True)

# define training points over the entire domain, for the physics loss
# TODO: write code here
x_physics = torch.linspace(0,6000,30).view(-1,1).requires_grad_(True)

# train the PINN
x_test = torch.linspace(0,6000,300).view(-1,1)
u_exact = exact_solution(L, E, I, q, x_test)
optimiser = torch.optim.Adam(pinn.parameters(),lr=1e-3)

iteration = 15001
# Initialize arrays to store losses
losses1 = np.zeros(iteration)
losses2 = np.zeros(iteration)
losses3 = np.zeros(iteration)
losses4 = np.zeros(iteration)
losses5 = np.zeros(iteration)

for i in range(iteration):
    optimiser.zero_grad()

    # compute boundary loss at x = 0
    u = pinn(x_bc0) # (1,1)
    loss1 = (torch.squeeze(u) - 0)**2

    dudt = torch.autograd.grad(u, x_bc0, torch.ones_like(u), create_graph=True)[0]
    loss2 = (torch.squeeze(dudt) - 0)**2

    # compute boundary loss at x = L
    u = pinn(x_bcL) # (1,1)
    loss3 = (torch.squeeze(u) - 0)**2

    dudt = torch.autograd.grad(u, x_bcL, torch.ones_like(u), create_graph=True)[0]
    loss4 = (torch.squeeze(dudt) - 0)**2
    
    # compute physics loss
    u = pinn(x_physics)
    dudt = torch.autograd.grad(u, x_physics, torch.ones_like(u), create_graph=True)[0]
    d2udt2 = torch.autograd.grad(dudt, x_physics, torch.ones_like(dudt), create_graph=True)[0]
    d3udt3 = torch.autograd.grad(d2udt2, x_physics, torch.ones_like(dudt), create_graph=True)[0]
    d4udt4 = torch.autograd.grad(d3udt3, x_physics, torch.ones_like(dudt), create_graph=True)[0]
    loss5 = torch.mean((E*I*d4udt4 - q )**2)

    # Store the losses
    losses1[i] = loss1.item()
    losses2[i] = loss2.item()
    losses3[i] = loss3.item()
    losses4[i] = loss4.item()
    losses5[i] = loss5.item()
    
    # backpropagate joint loss, take optimiser step
    loss = loss1 + loss2 + loss3 + loss4 + loss5*10
    loss.backward()
    optimiser.step()

    # plot the result as training progresses
    if i % 5000 == 0:
        #print(u.abs().mean().item(), dudt.abs().mean().item(), d2udt2.abs().mean().item())
        u = pinn(x_test).detach()
        plt.figure(figsize=(6,2.5))
        plt.scatter(x_physics.detach()[:,0],
                    torch.zeros_like(x_physics)[:,0], s=20, lw=0, color="tab:green", alpha=0.6)
        plt.plot(x_test[:,0], u_exact[:,0], label="Exact solution", color="tab:grey", alpha=0.6)
        plt.plot(x_test[:,0], u[:,0], label="PINN solution", color="tab:green")
        plt.title(f"Training step {i}")
        plt.legend()
        plt.show()
#%%
# Plot the losses
plt.figure(figsize=(10, 6))
plt.plot(losses1, label='Loss 1')
plt.plot(losses2, label='Loss 2')
plt.plot(losses3, label='Loss 3')
plt.plot(losses4, label='Loss 4')
plt.plot(losses5, label='Loss 5')

# Add plot details
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Comparison')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()