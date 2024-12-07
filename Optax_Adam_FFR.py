import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt

# Constants
T = 12.0  # time horizon
N = 120  # number of control intervals
dt = T / N  # time step
alpha = 1 # constant
beta = 1 # another constant

# Initial conditions and target
x0 = jnp.array([-10, -10, jnp.pi/2, 0, 0, 0])  # initial state
x_target = jnp.array([0, 0, 0, 0, 0, 0])  # target state

# Function to integrate the next state
def next_state(Xk, Uk, dt):
    x1, x2, x3, x4, x5, x6 = Xk
    u1, u2 = Uk
    # Dynamics
    x1_dot = x4
    x2_dot = x5
    x3_dot = x6
    x4_dot = (u1 + u2)*jnp.cos(x3)
    x5_dot = (u1 + u2)*jnp.sin(x3)
    x6_dot = alpha*u1 - beta*u2
    return Xk + dt * jnp.array([x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot])

# Define the objective function for optimization
def objective(U, X0, dt, x_target):
    X = jnp.zeros((6, N + 1))
    X = X.at[:, 0].set(X0)
    J = 0.0

    for idx in range(N):
        Uk = U[idx]
        X = X.at[:, idx + 1].set(next_state(X[:, idx], Uk, dt))
        # Calculate stage cost with control penalties and target deviation
        L = Uk[0]**2 + Uk[1]**2
        J += L * dt

    # Terminal cost to penalize deviation from the target state
    terminal_cost = jnp.sum(jnp.square(X[:, -1] - x_target))
    return J + 10 * terminal_cost

# Projection function for constraints
"""
def project_constraints(U, X0, dt):
    X = jnp.zeros((6, N + 1))
    X = X.at[:, 0].set(X0)

    for idx, Uk in enumerate(U):
        X = X.at[:, idx + 1].set(next_state(X[:, idx], Uk, dt))

    # Enforce the lower bound for x6 indirectly by adjusting u2
    U_adjusted = [(u[0], jnp.where(X[5, idx + 1] < x6_lb, 0, u[1])) for idx, u in enumerate(U)]
    return U_adjusted
"""

# Initial guess for controls (non-zero random guess)
key = jax.random.PRNGKey(0)
U0 = [(jax.random.uniform(key, minval=-0.5, maxval=0.5), jax.random.uniform(key, minval=-0.5, maxval=0.5)) for _ in range(N)]

# Optimizer setup
learning_rate = 0.01
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(U0)

@jax.jit
def step(U, opt_state):
    loss, grads = jax.value_and_grad(objective)(U, x0, dt, x_target)
    updates, opt_state = optimizer.update(grads, opt_state)
    U = optax.apply_updates(U, updates)
    return U, opt_state, loss

# Optimization loop
# U = U0
# num_iterations = 10000
# loss_history = []
# for i in range(num_iterations):
#     U, opt_state, loss = step(U, opt_state)
#     loss_history.append(loss)
#     if i % 100 == 0:
#         print(f"Iteration {i}, Loss: {loss}")

# Optimization loop
U = U0
max_iter = 100000
loss_history = [jnp.inf]
for i in range(max_iter):
    U, opt_state, loss = step(U, opt_state)
    loss_history.append(loss)
    if i % 10000 == 0:
        print(f"Iteration {i}, Loss: {loss}")
    if jnp.abs(loss_history[-1] - loss_history[-2])/loss_history[-1] < 1e-8:
      break

# Simulate the system with the optimal control
x1_sol = [x0[0]]
x2_sol = [x0[1]]
x3_sol = [x0[2]]
x4_sol = [x0[3]]
x5_sol = [x0[4]]
x6_sol = [x0[5]]
xk = x0
for uk in U:
    xk = next_state(xk, uk, dt)
    x1_sol.append(xk[0])
    x2_sol.append(xk[1])
    x3_sol.append(xk[2])
    x4_sol.append(xk[3])
    x5_sol.append(xk[4])
    x6_sol.append(xk[5])

# Plot the results
t_grid = jnp.linspace(0, T, N + 1)
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t_grid, x1_sol, label='x1')
plt.plot(t_grid, x2_sol, label='x2')
plt.plot(t_grid, x3_sol, label='x3')
plt.plot(t_grid, x4_sol, label='x4')
plt.plot(t_grid, x5_sol, label='x5')
plt.plot(t_grid, x6_sol, label='x6')
plt.xlabel('Time')
plt.ylabel('States')
plt.legend()

plt.subplot(3, 1, 2)
plt.step(t_grid[:-1], [u[0] for u in U], label='u1', where='post')
plt.step(t_grid[:-1], [u[1] for u in U], label='u2', where='post')
plt.xlabel('Time')
plt.ylabel('Control input')
plt.legend()

plt.tight_layout()
plt.show()
