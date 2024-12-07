from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import numpy as np
import matplotlib.pyplot as plt

# Constants
T = 10.0  # time horizon
N = 100  # number of control intervals
dt = T / N  # time step
x2_lb = -0.25  # lower bound for x2

# Initial conditions
x0 = np.array([0.0, 1.0])  # initial state (x1(0)=0, x2(0)=1)

# Function to integrate the next state
def next_state(Xk, Uk, dt):

    x1, x2 = Xk
    x1_dot = (1 - x2**2) * x1 - x2 + Uk  # Dynamics
    x2_dot = x1
    return Xk + dt * np.array([x1_dot, x2_dot])

# Define the objective function for optimization
def objective(U, X0, dt):

    X = np.zeros((2, len(U)+1))
    X[:, 0] = X0

    L = np.zeros(len(U))

    J = np.zeros(len(U))

    for idx, Uk in enumerate(U):
        X[:, idx+1] = next_state(X[:, idx], Uk, dt)
        L[idx] = X[0, idx+1]**2 + X[1, idx+1]**2 + Uk**2
        J[idx] = L[idx] * dt

    return np.sum(J)

# Initial guess for controls (non-zero guess)
U0 = np.full(N, np.random.rand(1))  # Starting control inputs set to random

ctrl_con = LinearConstraint(np.eye(N), lb=-1, ub=1)
spc_con = NonlinearConstraint(lambda U: space_con(U, x0, dt), lb=x2_lb, ub=np.inf)

def space_con(U, X0, dt):

    X = np.zeros((2, len(U)+1))
    X[:, 0] = X0

    for idx, Uk in enumerate(U):
        X[:, idx+1] = next_state(X[:, idx], Uk, dt)

    return X[1, :]


# Minimize the objective function
result = minimize(fun=lambda U_opt: objective(U_opt, x0, dt),
                  x0=U0,
                  constraints=[ctrl_con, spc_con],
                  method='SLSQP',
                  options={'disp': True})

# Extract optimal control values
U_opt = result.x

# Simulate the system with the optimal control
x1_sol = [x0[0]]
x2_sol = [x0[1]]
xk = np.array(x0)
for uk in U_opt:
    xk = next_state(xk, uk, dt)
    x1_sol.append(xk[0])
    x2_sol.append(xk[1])

# Plot the results
t_grid = np.linspace(0, T, N + 1)
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t_grid, x1_sol, label='x1')
plt.plot(t_grid, x2_sol, label='x2')
plt.axhline(y=x2_lb, color='r', linestyle='--', label='x2 lower bound')
plt.xlabel('Time')
plt.ylabel('States')
plt.legend()

plt.subplot(3, 1, 2)
plt.step(t_grid[:-1], U_opt, label='u', where='post')
plt.xlabel('Time')
plt.ylabel('Control input')
plt.legend()

plt.tight_layout()
plt.show()
