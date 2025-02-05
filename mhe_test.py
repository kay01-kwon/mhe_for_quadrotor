import numpy as np
from ode_solver import custom_rk4 as rk4
from model.dynamics_model import DynamicsModel
from scipy.integrate import odeint
from ocp_solver.mhe_ocp_solver import MheOcpSolver
import matplotlib.pyplot as plt
# from matplotlib import animation
# from matplotlib.animation import FuncAnimation


if __name__ == '__main__':

    s = np.array([1, 0, 0, 0, 0, 0, 0])

    num_var = s.shape[0]

    T_sim = 10
    dt = 0.01
    tspan = np.arange(0, T_sim, dt)

    s_array = np.zeros((num_var, len(tspan)))
    s_array[:, 0] = s

    J_true = np.array([0.023, 0.026, 0.041])
    r_CM = np.array([0, 0, 0])
    # r_CM = np.array([-0.005, -0.00014, -0.069])


    # Test for roll
    u = np.array([0.01, -0.01, -0.01, 0.01])
    # Test for pitch
    # u = np.array([0.01, 0.01, -0.01, -0.01])
    # Test for yaw
    # u = np.array([0.01, -0.01, 0.01, -0.01])

    l = 0.330
    C_T = 1
    C_M = 1


    model_to_test = DynamicsModel(J_true, r_CM, l, C_T, C_M)


    roll_array = np.empty((len(tspan)))
    pitch_array = np.empty((len(tspan)))
    yaw_array = np.empty((len(tspan)))

    for i in range(len(tspan)):

        t_span_ode = [tspan[i], tspan[i]+dt]
        s_next = rk4.custom_rk4_solver(model_to_test.rotational_dynamics,
                                       s, u, tspan=t_span_ode)
        s = s_next
        s_array[:, i] = s

        qw = s[0]
        qx = s[1]
        qy = s[2]
        qz = s[3]

        roll_array[i] = np.arctan2(2*(qw*qx + qy*qz),
                                   1-2*(qx*qx + qy*qy))

        pitch_array[i] = -np.pi/2 + 2*np.arctan2(
                            np.sqrt(1+2*(qw*qy-qx*qz)),
                            np.sqrt(1-2*(qw*qy-qx*qz)))

        yaw_array[i] = np.arctan2(2*(qw*qz + qx*qy),
                                    1-2*(qy*qy + qz*qz))


    plt.subplot(4,1,1)
    plt.plot(tspan, s_array[0,:])

    plt.subplot(4,1,2)
    plt.plot(tspan, s_array[1,:])

    plt.subplot(4,1,3)
    plt.plot(tspan, s_array[2,:])

    plt.subplot(4,1,4)
    plt.plot(tspan, s_array[3,:])

    plt.show()