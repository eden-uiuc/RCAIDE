# Make sure to install CasADi using:
# pip install casadi

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

T = 10.0  # time horizon
N = 100  # number of control intervals
dt = T/N # time step

# Define state and control variables
x1 = ca.MX.sym('x1')  # state variable x1
x2 = ca.MX.sym('x2')  # state variable x2
u = ca.MX.sym('u')    # control input u

# Define the state dynamics
x1_dot = (1 - x2**2)*x1 - x2 + u  # x1_dot (dx1/dt)
x2_dot = x1                    # x2_dot (dx2/dt)

# Define the state and control vectors
X = ca.vertcat(x1, x2)
U = u

# Define the cost function integrand
L = x1**2 + x2**2 + u**2

# Build an integrator using CasADi
dae = {'x': X, 'p': U, 'ode': ca.vertcat(x1_dot, x2_dot), 'quad': L}
opts = {'tf': dt}
F = ca.integrator('F', 'cvodes', dae, opts)

# Initial conditions
x0 = [0, 1]  # initial state (x1(0)=0, x2(0)=1)
X0 = ca.MX(x0)

# Decision variables
X_opt = []
U_opt = []
J = 0  # Objective (cost function)

# Define bounds for the control input and states
u_lb = -1
u_ub = 1
x2_lb = -0.25

# Optimization variables
Xk = X0
for k in range(N):
    Uk = ca.MX.sym(f'U_{k}')
    U_opt.append(Uk)

    # Compute next state using the integrator
    Fk = F(x0=Xk, p=Uk)
    Xk_next = Fk['xf']
    Lk = Fk['qf']

    # Append next state to the list
    X_opt.append(Xk_next)

    # Update the cost function
    J += Lk

    # Enforce state constraints
    g = []
    g.append(Xk_next[1] - x2_lb)  # x2 >= 0.25

    # Update current state for next iteration
    Xk = Xk_next

# Concatenate decision variables into vectors
U_opt = ca.vertcat(*U_opt)
g = ca.vertcat(*g)

# Create NLP solver
nlp = {'x': U_opt, 'f': J, 'g': g}
solver = ca.nlpsol('solver', 'ipopt', nlp)

# Solve the optimization problem
sol = solver(x0=np.zeros(N), lbx=u_lb, ubx=u_ub, lbg=0, ubg=ca.inf)

# Extract optimal control and states
u_opt = sol['x'].full().flatten()

# Simulate the system with the optimal control
x1_sol = [x0[0]]
x2_sol = [x0[1]]
xk = np.array(x0)
for uk in u_opt:
    Fk = F(x0=xk, p=uk)
    xk = Fk['xf'].full().flatten()
    x1_sol.append(xk[0])
    x2_sol.append(xk[1])

# Plot the results
t_grid = np.linspace(0, T, N+1)
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t_grid, x1_sol, label='x1')
plt.plot(t_grid, x2_sol, label='x2')
plt.xlabel('Time')
plt.ylabel('States')
plt.legend()

plt.subplot(3, 1, 2)
plt.step(t_grid[:-1], u_opt, label='u', where='post')
plt.xlabel('Time')
plt.ylabel('Control input')
plt.legend()

plt.show()
