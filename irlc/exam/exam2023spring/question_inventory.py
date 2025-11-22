from irlc.ex02.inventory import InventoryDPModel
from irlc.ex02.dp import DP_stochastic
from irlc.ex02.dp_model import DPModel
import numpy as np


class FlowerStoreModel(InventoryDPModel): #!b;nolines
    def __init__(self, N=3, c=0., prob_empty=False):
        self.c = c
        self.prob_empty = prob_empty
        super().__init__(N=N)

    def g(self, x, u, w, k):  # Cost function g_k(x,u,w)
        if self.prob_empty:
            return 0
        return u * self.c + np.abs(x + u - w)

    def f(self, x, u, w, k):  # Dynamics f_k(x,u,w)
        return max(0, min(max(self.S(k)), x + u - w))

    def Pw(self, x, u, k):  # Distribution over random disturbances
        pw = {0: .1, 1: .3, 2: .6}
        return pw

    def gN(self, x):
        if self.prob_empty:
            return -1 if x == 1 else 0
        else:
            return 0 #!b

def a_get_policy(N: int, c: float, x0 : int) -> int:
    model = FlowerStoreModel(N=N, c=c, prob_empty=False) #!b;nolines
    J, pi = DP_stochastic(model)
    u = pi[0][x0]  #!b
    return u

def b_prob_one(N : int, x0 : int) -> float:
    model = FlowerStoreModel(N=N, prob_empty=True) #!b;nolines
    J, pi = DP_stochastic(model)
    pr_empty = -J[0][x0] #!b
    return pr_empty


if __name__ == "__main__":
    model = InventoryDPModel()
    pi = [{s: 0 for s in model.S(k)} for k in range(model.N)]
    x0 = 0
    c = 0.5
    N = 3
    print(f"a) The policy choice for {c=} is {a_get_policy(N, c,x0)} should be 1")
    print(f"b) The probability of ending up with a single element in the inventory is {b_prob_one(N, x0)} and should be 0.492")
