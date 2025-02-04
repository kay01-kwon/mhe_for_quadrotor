import numpy as np
from scipy.integrate import odeint
from ocp_solver.mhe_ocp_solver import MheOcpSolver
import matplotlib.pyplot as plt
# from matplotlib import animation
# from matplotlib.animation import FuncAnimation

J_true = np.diag([0.023, 0.026, 0.041,])
r_CM = np.array([-0.005, -0.00014, -0.069])
