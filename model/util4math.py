import numpy as np
import casadi as cs

def quaternion2rotm(q):
    '''
    Convert quaternion to rotation matrix
    The argument and output should be casadi form.
    :param q: qw = q[0], qx = q[1], qy = q[2], qz = q[3]
    :return: rotm
    '''

    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    # Represent the return value as Casadi format
    rotm = cs.vertcat(
        cs.horzcat(1-2*(qy*qy + qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)),
        cs.horzcat(2*(qy*qx+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)),
        cs.horzcat(2*(qz*qx-qw*qy), 2*(qz*qy+qw*qx), 1-2*(qx*qx+qy*qy))
    )
    return rotm

def quat2quat_vec(q):
    '''
    Convert quaternion to quaternion vector
    :param q: qw = q[0], qx = q[1], qy = q[2], qz = q[3]
    :return: q_vec = [qx, qy, qz]
    '''

    qx = q[1]
    qy = q[2]
    qz = q[3]

    # Extract x, y, and z elements of quaternion
    q_vec = cs.vertcat(qx, qy, qz)
    return q_vec

def vec2skew_symmetric_matrix(v):
    '''
    Convert vector to skew symmetric matrix
    :param v: vx = v[0], vy = v[1], vz = v[2]
    :return: skew_symmetric_matrix
    '''

    vx = v[0]
    vy = v[1]
    vz = v[2]

    skew_symmetric_matrix = cs.vertcat(
        cs.horzcat(0.0, -vz, vy),
        cs.horzcat(vz, 0.0, -vx),
        cs.horzcat(-vy, vx, 0.0)
    )

def otimes(q1, q2):
    '''
    Compute the multiplication of two quaternions
    :param q1:
    :param q2:
    :return:
    '''

    q1_w = q1[0]
    q1_x = q1[1]
    q1_y = q1[2]
    q1_z = q1[3]

    # Left quaternion 4 x 4 matrix
    q1_L = cs.vertcat(
        cs.horzcat(q1_w, -q1_x, -q1_y, -q1_z),
        cs.horzcat(q1_x, q1_w, -q1_z, q1_y),
        cs.horzcat(q1_y, q1_z, q1_w, -q1_x),
        cs.horzcat(q1_z, -q1_y, q1_x, q1_w)
    )

    q1_otimes_q2 = cs.mtimes(q1_L, q2)

    return q1_otimes_q2

def thrust2FM(model_type, thrust, arm_length, C_T, C_M):
    '''
    Compute the force and moment from four rotors' thrusts
    :param model_type: '+', 'x'
    :param thrust: four rotors' thrust
    :param arm_length: arm length
    :param C_T: Thrust coefficient
    :param C_M: Moment coefficient
    :return: m_x, m_y, m_z
    '''

    if model_type == '+':
        l = arm_length
        m_x = l*( thrust[1] - thrust[3])
        m_y = l*( thrust[2] - thrust[0])
    else:
        l = arm_length*np.sqrt(2)/2
        m_x = l*( thrust[0] - thrust[1] - thrust[2] + thrust[3])
        m_y = l*( (thrust[0] + thrust[1]) - (thrust[2] + thrust[3]))

    m_z = C_M/C_T*( thrust[0] - thrust[1] + thrust[2] - thrust[3] )

    return m_x, m_y, m_z