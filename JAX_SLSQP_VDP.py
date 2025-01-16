import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import matplotlib.pyplot as plt

# Set JAX to use 64-bit precision
jax.config.update("jax_enable_x64", True)

# Constants
T = 10.0  # Time horizon
N = 100   # Number of control intervals
dt = T/N  # Time step
x2_lb = -0.25  # Lower bound for x2

# Initial conditions
x0 = jnp.array([0.0, 1.0], dtype=jnp.float64)  # Initial state

# Function to integrate the next state
def next_state(Xk, Uk, dt):
    x1, x2 = Xk
    x1_dot = (1 - x2**2)*x1 - x2 + Uk  # Dynamics
    x2_dot = x1
    return Xk + dt*jnp.array([x1_dot, x2_dot], dtype=jnp.float64)

@jax.jit
def objective(U, X0, dt):
    X = jnp.zeros((2, len(U)+1), dtype=jnp.float64)
    X = X.at[:,0].set(X0)  # Initial state

    L = jnp.zeros(len(U), dtype=jnp.float64)  # Lagrange term of integrand
    J = 0.0  # Cost function

    for idx, Uk in enumerate(U):
        X = X.at[:,idx+1].set(next_state(X[:,idx], Uk, dt))
        L = X[0,idx+1]**2 + X[1,idx+1]**2 + Uk**2
        J += L*dt

    return J

@jax.jit
def space_constraint(U, X0, dt):
    X = jnp.zeros((2, len(U)+1), dtype=jnp.float64)
    X = X.at[:,0].set(X0)  # Initial state

    for idx, Uk in enumerate(U):
        X = X.at[:,idx+1].set(next_state(X[:,idx], Uk, dt))

    return X[1,:]

# Initialize control input
key = jax.random.PRNGKey(0)
U0 = jax.random.uniform(key, shape=(N,), dtype=jnp.float64, minval=-1, maxval=1)

# Define constraints
ctrl_constraint = LinearConstraint(np.eye(N), lb=-1, ub=1)
spc_constraint = NonlinearConstraint(lambda U: np.array(space_constraint(U, x0, dt)), lb=x2_lb, ub=np.inf)

# Minimize the objective function
res = minimize(
    fun=lambda U_opt: float(objective(U_opt, x0, dt)),
    x0=U0,
    method='SLSQP',
    constraints=[spc_constraint, ctrl_constraint],
    options={'disp': True}
)

U_opt = res.x

# Simulate the system with the optimal control
X = jnp.zeros((2, N+1), dtype=jnp.float64)
X = X.at[:,0].set(x0)

for idx, Uk in enumerate(U_opt):
    X = X.at[:,idx+1].set(next_state(X[:,idx], Uk, dt))

# Plot the results
t_grid = np.linspace(0, T, N + 1)
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t_grid, X[0,:], label='x1')
plt.plot(t_grid, X[1,:], label='x2')
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
