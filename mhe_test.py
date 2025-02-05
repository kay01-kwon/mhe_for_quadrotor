import numpy as np
from scipy.integrate import odeint
from ocp_solver.mhe_ocp_solver import MheOcpSolver
import matplotlib.pyplot as plt
# from matplotlib import animation
# from matplotlib.animation import FuncAnimation

J_true = np.diag([0.023, 0.026, 0.041,])
r_CM = np.array([-0.005, -0.00014, -0.069])

def custom_rk4_solver(f, s, u, tspan):
    '''
    Custom RK4
    :param f: function of dynamics
    :param s: state
    :param u: input
    :param tspan: t0 and t1
    :return: s_next
    '''
    dt = tspan[1] - tspan[0]
    tm = tspan[0] + dt/2
    tf = tspan[1]
    K1 = f(tspan[0], s, u)
    K2 = f(tm, s + K1*dt/2, u)
    K3 = f(tm, s + K2*dt/2, u)
    K4 = f(tf, s + K3*dt, u)
    s_next = s + 1/6*(K1 + 2*K2 + 2*K3 + K4)*dt
    return s_next

m = 1
def dynamics(t, s, F):
    A = np.array([[0, 1],[0, 0]])
    B = np.array([0, 1])

    u = F/m
    dsdt = A@s + B*u
    return dsdt

if __name__ == '__main__':

    s = np.array([0, 0])

    T_sim = 10
    dt = 0.01
    tspan = np.arange(0, T_sim, dt)

    s_array = np.zeros((2, len(tspan)))
    s_array[:, 0] = s

    u = -1

    for i in range(len(tspan)):

        t_span_ode = [tspan[i], tspan[i]+dt]
        s_next = custom_rk4_solver(dynamics, s, u, tspan=t_span_ode)
        s = s_next

        s_array[:, i] = s


    plt.plot(tspan, s_array[0,:])
    plt.show()