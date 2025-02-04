import numpy as np
import scipy.linalg
from acados_template import AcadosOcpSolver, AcadosOcp
from model.mhe_model import MheQuadrotorModel
import casadi as cs

class MheOcpSolver:
    def __init__(self, dt, tf, R, Q0, Q, num_param):
        '''
        :param dt: sampling time step
        :param tf: prediction horizon
        :param R: Measurement noise
        :param Q0: Weight for arrival cost
        :param Q: Weight for state evolution
        '''

        # Prediction horizon
        self.tf = tf

        # Sampling time step
        self.dt = dt

        # The number of nodes
        N = tf/dt

        # Generate AcadosOcp
        self.ocp_mhe = AcadosOcp()

        self.acados_mhe_solver = []

        # Setup for ocp_mhe (AcadosOcp object)

        # 1. Pass model to ocp_mhe object
        self.ocp_mhe.model = MheQuadrotorModel().get_acados_model()

        # 2. set the number of multiple shooting nodes
        self.ocp_mhe.dims.N = N

        # Get dimension info from model
        self.nx_aug = self.ocp_mhe.model.x.rows()
        self.nx = self.nx_aug - num_param
        self.nu = self.ocp_mhe.model.u.rows()
        self.ny_0 = R.shap[0] + Q.shape[0] + Q0.shape[0]    # h(x), w, and arrival cost
        self.ny = R.shape[0] + Q.shape[0]                   # h(x), w
        self.ny_e = 0
        self.nparam = self.ocp_mhe.model.p.rows()

        # Pass state and control variables
        self.x = self.ocp_mhe.model.x
        self.u = self.ocp_mhe.model.u

        # set cost type
        self.ocp_mhe.cost.cost_type = 'NONLINEAR_LS'
        self.ocp_mhe.cost.cost_type_e = 'LINEAR_LS'
        self.ocp_mhe.cost.cost_type_0 = 'NONLINEAR_LS'

        self.ocp_mhe.parameter_values = np.zeros((self.nparam,))

        self.set_ocp_cost(Q0, Q, R)

        self.set_ocp_solver()

    def set_ocp_cost(self, Q0, Q, R):
        # Setup weight for cost

        self.ocp_mhe.cost.W_0 = scipy.linalg.block_diag(R, Q, Q0)
        self.ocp_mhe.model.cost_y_expr_0 = cs.vertcat(self.x[:self.nx],
                                                      self.u,
                                                      self.x)
        self.ocp_mhe.cost.yref_0 = np.zeros((self.ny_0,))

        # 4. Weight for intermediate cost
        self.ocp_mhe.cost.W = scipy.linalg.block_diag(R, Q)
        self.ocp_mhe.model.cost_y_expr = cs.vertcat(self.x[:self.nx],
                                                    self.u)
        self.ocp_mhe.cost.yref = np.zeros((self.ny,))

    def set_ocp_solver(self):

        # Set QP solver
        self.ocp_mhe.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        self.ocp_mhe.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp_mhe.solver_options.integrator_type = 'ERK'

        self.ocp_mhe.solver_options.nlp_solver_type = 'SQP'
        self.ocp_mhe.solver_options.nlp_solver_max_iter = 200

        # Set prediction horizon
        self.ocp_mhe.solver_options.tf = self.tf

        self.acados_mhe_solver = AcadosOcpSolver(self.ocp_mhe)

    def get_ocp_solver(self):

        return self.acados_mhe_solver