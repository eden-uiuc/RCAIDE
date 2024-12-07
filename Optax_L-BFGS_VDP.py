import jax
import jax.numpy as jnp
import optax
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
def objective(U):
    X = jnp.zeros((2, len(U) + 1))
    X = X.at[:, 0].set(x0)
    J = 0.0

    for idx, Uk in enumerate(U):
        X = X.at[:, idx + 1].set(next_state(X[:, idx], Uk, dt))
        L = X[0, idx + 1]**2 + X[1, idx + 1]**2 + Uk**2
        J += L * dt

    return J

# Projection function for constraints
def project_constraints(U):
    # Linear control constraint: -1 <= U <= 1
    U = jnp.clip(U, -1, 1)

    # Simulate to check state constraints
    X = jnp.zeros((2, len(U)+1))
    X = X.at[:, 0].set(x0)
    for idx, Uk in enumerate(U):
        X = X.at[:, idx+1].set(next_state(X[:, idx], Uk, dt))

    # Adjust x2 where it violates the lower bound
    x2_violations = jnp.where(X[1, :] < x2_lb, x2_lb, X[1, :])
    X = X.at[1, :].set(x2_violations)
    return U

# Initial guess for controls
key = jax.random.PRNGKey(0)
U0 = jax.random.uniform(key, (N,), minval=-0.5, maxval=0.5)

# Optimization setup
@jax.jit
def optimize_step(U, opt_state):
    # Compute value and gradient
    value_and_grad = jax.value_and_grad(objective)
    value, grad = value_and_grad(U)

    # Update with L-BFGS
    updates, new_opt_state = optimizer.update(
        grad,
        opt_state,
        params=U,
        value=value,
        grad=grad,
        value_fn=objective
    )
    new_U = U + updates

    # Project constraints
    new_U = project_constraints(new_U)

    return new_U, new_opt_state, value

# Optimizer setup
optimizer = optax.lbfgs()
opt_state = optimizer.init(U0)

# Optimization loop
U = U0
num_iterations = 1000
loss_history = []

for i in range(num_iterations):
    U, opt_state, loss = optimize_step(U, opt_state)
    loss_history.append(loss)
    if i % 50 == 0:
        print(f"Iteration {i}, Loss: {loss:.6f}")

# Simulate the system with the optimal control
x1_sol = jnp.zeros(N + 1)
x2_sol = jnp.zeros(N + 1)
x1_sol = x1_sol.at[0].set(x0[0])
x2_sol = x2_sol.at[0].set(x0[1])
xk = x0

for i, uk in enumerate(U):
    xk = next_state(xk, uk, dt)
    x1_sol = x1_sol.at[i + 1].set(xk[0])
    x2_sol = x2_sol.at[i + 1].set(xk[1])

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

plt.subplot(3, 1, 3)
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()

print(f"Final Loss: {loss_history[-1]}")
