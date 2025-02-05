from acados_template import AcadosModel
import casadi as cs
from model import util4math

class MheQuadrotorModel:
    def __init__(self, model_type, l, C_T, C_M):
        '''
        Attitude dynamics model
        dwdt = J_inv*(M - r_CMxf - wxJw)
        Estimate J_true, r_CM, and sigma!
        '''
        self.model_name = "mhe_attitdue_model"
        self.model_type = model_type
        self.l = l
        self.C_T = C_T
        self.C_M = C_M

        self.acados_model = AcadosModel()

        # State variables declaration
        self.q = cs.MX.sym('q', 4)              # quaternion
        self.w = cs.MX.sym('w', 3)              # angular velocity
        self.J = cs.MX.sym('J', 3)              # Moment of inertia
        self.r_CM = cs.MX.sym('r_CM',2)         # COM offset
        self.x = cs.vertcat(self.q, self.w, self.J, self.r_CM)

        # Noise : quaternion and angular velocities
        self.w_q = cs.MX.sym('w_q', 1)
        self.w_w = cs.MX.sym('w_w', 1)
        self.w_noise = cs.vertcat(self.w_q, self.w_w)

        # Control input declartaion (Pararmeter to pass)
        self.u1 = cs.MX.sym('u1')
        self.u2 = cs.MX.sym('u2')
        self.u3 = cs.MX.sym('u3')
        self.u4 = cs.MX.sym('u4')
        self.u = cs.vertcat(self.u1, self.u2, self.u3, self.u4)

        # dxdt declaration
        self.dqdt = cs.MX.sym('dqdt',4)
        self.dwdt = cs.MX.sym('dwdt',3)
        self.dJdt = cs.MX.sym('dJdt',3)
        self.dr_CMdt = cs.MX.sym('dr_CMdt',2)
        self.xdot = cs.vertcat(self.dqdt, self.dwdt,
                               self.dJdt, self.dr_CMdt)


    def get_acados_model(self):

        self.f_expl = cs.vertcat(self.q_kinematics(), self.w_dynamics(),
                                 self.J_dynamics(), self.COM_dynamics())

        #
        self.f_expl = self.f_expl + self.w_noise
        self.f_impl = self.xdot

        # Add state noise
        self.acados_model.f_expl_expr = self.f_expl
        self.acados_model.f_impl_expr = self.f_impl - self.f_expl

        self.acados_model.x = self.x
        self.acados_model.u = self.w_noise
        self.acados_model.xdot = self.xdot
        self.acados_model.p = self.u

        self.acados_model.name = self.model_name

        return self.acados_model

    def q_kinematics(self):
        '''
        Compute the rotational kinematics
        :return: dqdt
        '''
        w_quat = cs.vertcat(self.w)
        dqdt = util4math.otimes(self.q, w_quat)
        return dqdt

    def w_dynamics(self):
        '''
        Compute the rotational dynamics
        :return: dwdt
        '''

        # MOI
        Jxx = self.J[0]
        Jyy = self.J[1]
        Jzz = self.J[2]

        # COM offset
        rx = self.r_CM[0]
        ry = self.r_CM[1]

        # Angular velocity
        w_x = self.w[0]
        w_y = self.w[1]
        w_z = self.w[2]

        M_x, M_y, M_z = util4math.thrust2FM(self.model_type,
                                            self.u,
                                            self.C_T,
                                            self.C_M)

        # Moment divided by MOI
        M_vec = cs.vertcat(M_x/Jxx, M_y/Jyy, M_z/Jzz)

        # Collective thrust
        f_col = self.u[0] + self.u[1] + self.u[2] + self.u[3]

        # Moment induced by collective thrust
        r_cross_f_col = cs.vertcat(ry*f_col,
                                   -rx*f_col,
                                   0)

        # inertial effect = w x (J*w)
        inertial_effect = cs.vertcat((Jzz-Jyy)/Jxx*w_y*w_z,
                                     (Jxx-Jzz)/Jyy*w_x*w_z,
                                     (Jyy-Jxx)/Jzz*w_x*w_y)

        dwdt = M_vec - r_cross_f_col - inertial_effect
        return dwdt

    def J_dynamics(self):
        return cs.vertcat(0, 0, 0)

    def COM_dynamics(self):
        return cs.vertcat(0, 0)