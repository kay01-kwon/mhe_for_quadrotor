import numpy as np

m = 1

class DynamicsModel:
    def __init__(self, J_true, r_CM, l, C_T, C_M):
        # Initialize parameter
        self.m = m
        self.J_true = J_true
        self.r_CM = r_CM
        self.l = l
        self.C_T = C_T
        self.C_M = C_M
        self.FD = np.zeros((4,4))

        self.forward_dynamics()

    def forward_dynamics(self):

        l = self.l
        C_T = self.C_T
        C_M = self.C_M

        self.FD = np.array([
            [1, 1, 1, 1],
            [l, -l, -l, l],
            [l, l, -l, -l],
            [C_M/C_T, -C_M/C_T, C_M/C_T, -C_M/C_T]
                            ])

    def test_dynamics(self, t, s, F):
        A = np.array([[0, 1], [0, 0]])
        B = np.array([0, 1])

        u = F / self.m
        dsdt = A @ s + B * u
        return dsdt

    def rotational_dynamics(self, t, s, thrust):
        # state info
        # s[0] ~ s[3] : qw, qx, qy, qz
        # s[4] ~ s[6] : wx, wy, wz

        # MOI parameter
        Jxx = self.J_true[0]
        Jyy = self.J_true[1]
        Jzz = self.J_true[2]

        # COM parameter
        rx = self.r_CM[0]
        ry = self.r_CM[1]

        # Convert rotors' thrust into force and moment
        FM = self.FD @ thrust

        f = FM[0]
        m_vec = np.array([FM[1]/Jxx, FM[2]/Jyy, FM[3]/Jzz])
        print(m_vec)

        # quaternion and angular velocity
        q = s[0:4]
        w = s[4:]

        wx = w[0]
        wy = w[1]
        wz = w[2]

        # inertial effect = w x (J*w)
        inertial_effect = np.array([(Jzz-Jyy)/Jxx * wy * wz,
                                    (Jxx-Jzz)/Jyy * wx * wz,
                                    (Jyy-Jxx)/Jzz * wx * wy])

        # COM offset effect
        r_cross_f = np.array([ry*f,
                              -rx*f,
                              0])

        dwdt = m_vec - inertial_effect - r_cross_f

        w_quat_form = np.array([0, wx, wy, wz])
        dqdt = 0.5*self.otimes(q,w_quat_form)

        dsdt = np.hstack((dqdt, dwdt))

        return dsdt

    def otimes(self, q1, q2):
        '''
        Compute the multiplication of two quaternions
        :param q1: Left quaternion
        :param q2: Right quaternion
        :return: The multiplication of two quaternions
        '''
        den = np.sqrt(q1[0]*q1[0]
                      +q1[1]*q1[1]
                      +q1[2]*q1[2]
                      +q1[3]*q1[3])

        q1_w = q1[0]/den
        q1_x = q1[1]/den
        q1_y = q1[2]/den
        q1_z = q1[3]/den

        # Left quaternion 4 x 4 matrix
        q1_L = np.array([[q1_w, -q1_x, -q1_y, -q1_z],
                         [q1_x, q1_w, -q1_z, q1_y],
                         [q1_y, q1_z, q1_w, -q1_x],
                         [q1_z, -q1_y, q1_x, q1_w]])

        return q1_L @ q2