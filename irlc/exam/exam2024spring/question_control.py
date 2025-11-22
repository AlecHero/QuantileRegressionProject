import numpy as np
import sympy as sym
from irlc.ex03.control_model import ControlModel
from irlc.ex03.control_cost import SymbolicQRCost
from gymnasium.spaces import Box

class CustomModel(ControlModel):
    def __init__(self, u0, tF):
        super().__init__()
        self.u0 = u0
        self.tF = tF
        
    def sym_f(self, x, u, t=None):
        xdot = -np.exp(self.u0-x[0]**2)
        return xdot


def a_xdot(x : float, a : float) -> float:
    u = lambda x,a: a*x**2
    xdot = lambda x,u: -np.exp(u-x**2)
    return xdot(x, u(x,a))

def b_rk4_simulate(u0 : float, tF : float):
    cmodel = CustomModel(u0, tF)
    cmodel.simulate(x0=0, u_fun=u0, t0=0, tF=tF)
    return 0#x

if __name__ == "__main__":
    print(f"a): dx/dt should be -1, you got {a_xdot(x=2, a=1)=}")
    print(f"b): Final position x(tF) should be approximately -2.09, you got {b_rk4_simulate(u0=2, tF=3)=}")
