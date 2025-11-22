from irlc.ex04.model_pendulum import PendulumModel
from irlc.ex04.discrete_control_model import DiscreteControlModel
from irlc.ex06.dlqr import LQR
import numpy as np


def a_LQR_solve(a : float, x0 : np.ndarray) -> float:
    A = np.asarray([[1,a], [0, 1]])
    B = np.asarray([0, 1])[:,np.newaxis]
    d = np.asarray([1,0])
    Q = np.eye(2)
    R = np.eye(1)
    N = 100
    
    (L,l),_ = LQR(A=[A]*N, B=[B]*N, d=[d]*N, Q=[Q]*N, R=[R]*N)
    u = L[0] @ x0 + l[0]
    return u[0]

def b_linearize(theta : float):
    model = PendulumModel()
    dmodel = DiscreteControlModel(model=model, dt=0.5)
    xbar = np.array([theta, 0])
    ubar = np.array([0])
    xp = dmodel.f(xbar, ubar)
    A, B = dmodel.f_jacobian(xbar, ubar)
    d = xp - A @ xbar - B @ ubar
    return A, B, d


def c_get_optimal_linear_policy(x0 : np.ndarray) -> float:
    A, B, d = b_linearize(theta=0)
    Q, R = np.eye(2), np.eye(1)
    N = 100
    (L,l),_ = LQR(A=[A]*N, B=[B]*N, d=[d]*N, Q=[Q]*N, R=[R]*N) 
    u = L[0] @ x0 + l[0]
    return u[0]

if __name__ == "__main__":
    theta = np.pi/2  # An example: linearize around theta = pi/2.
    a = 1
    x0 = np.asarray([1, 0])
    print(f"a) LQR action should be approximately -1.666, you got: {a_LQR_solve(a, x0)=}")
    A, B, d = b_linearize(theta) # Get the three matrices.
    print(f"b) Entry d[1] should be approx. 4.91, you got: {d[1]=}")
    theta = 0.1  # Try a small initial angle.
    print(f"c) Optimal policy for linearized problem should be approximately -1.07, you got: {c_get_optimal_linear_policy(x0=np.asarray([theta, 0]))=}")
