import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt

# Constants
T = 10.0  # time horizon
N = 100  # number of control intervals
dt = T / N  # time step
x2_lb = -0.25  # lower bound for x2

# Initial conditions
x0 = jnp.array([0.0, 1.0])  # initial state (x1(0)=0, x2(0)=1)

# Function to integrate the next state
def next_state(Xk, Uk, dt):
    x1, x2 = Xk
    x1_dot = (1 - x2**2) * x1 - x2 + Uk  # Dynamics
    x2_dot = x1
    return Xk + dt * jnp.array([x1_dot, x2_dot])

# Define the objective function for optimization
def objective(U, X0, dt):
    X = jnp.zeros((2, len(U)+1))
    X = X.at[:, 0].set(X0)
    J = 0.0

    for idx, Uk in enumerate(U):
        X = X.at[:, idx+1].set(next_state(X[:, idx], Uk, dt))
        L = X[0, idx+1]**2 + X[1, idx+1]**2 + Uk**2
        J += L * dt

    return J

# Projection function for constraints
def project_constraints(U, X0, dt):
    # Linear control constraint: -1 <= U <= 1
    U = jnp.clip(U, -1, 1) # Clips the control at each step

    # Nonlinear state constraint: x2 >= x2_lb
    X = jnp.zeros((2, len(U)+1))
    X = X.at[:, 0].set(X0)
    for idx, Uk in enumerate(U):
        X = X.at[:, idx+1].set(next_state(X[:, idx], Uk, dt))

    # Adjust x2 where it violates the lower bound
    x2_violations = jnp.where(X[1, :] < x2_lb, x2_lb, X[1, :])
    X = X.at[1, :].set(x2_violations)
    return U, X

# Initial guess for controls (non-zero)
key = jax.random.PRNGKey(0)
U0 = jax.random.uniform(key, (N,), minval=-0.5, maxval=0.5)

# Optimizer setup
learning_rate = 0.01 # Suitable range between 0.02-0.005
optimizer = optax.adam(learning_rate) # Using the Adam optimizer
opt_state = optimizer.init(U0) # Initializes the optimizer state

@jax.jit
def step(U, opt_state):
    loss, grads = jax.value_and_grad(objective)(U, x0, dt) # Calculate cost function and gradient
    updates, opt_state = optimizer.update(grads, opt_state) # Update parameters using optimizer
    U = optax.apply_updates(U, updates) # Apply updates to vector U
    
    # Project onto constraints
    U, _ = project_constraints(U, x0, dt) # Enforces the constraints on U
    
    return U, opt_state, loss

# Optimization loop
U = U0 # Initializing the control vector
num_iterations = 10000 # Number of iterations (needs to be >7500 for good results)
loss_history = [] # Initialize list to store loss history over time
for i in range(num_iterations): # Calculate loss history over time and store it
    U, opt_state, loss = step(U, opt_state)
    loss_history.append(loss)

# Simulate the system with the optimal control
x1_sol = [x0[0]]
x2_sol = [x0[1]]
xk = x0
for uk in U:
    xk = next_state(xk, uk, dt)
    x1_sol.append(xk[0])
    x2_sol.append(xk[1])

# Plot the results
t_grid = jnp.linspace(0, T, N + 1)
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t_grid, x1_sol, label='x1')
plt.plot(t_grid, x2_sol, label='x2')
plt.axhline(y=x2_lb, color='r', linestyle='--', label='x2 lower bound')
plt.xlabel('Time')
plt.ylabel('States')
plt.legend()

plt.subplot(3, 1, 2)
plt.step(t_grid[:-1], U, label='u', where='post')
plt.xlabel('Time')
plt.ylabel('Control input')
plt.legend()

plt.tight_layout()
plt.show()
