# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(''))))

import math
from irlc.exam.exam2024spring.inventory import InventoryDPModel
from irlc.exam.exam2024spring.dp import DP_stochastic

class BridalInventory(InventoryDPModel):
    def __init__(self, N=3, m=2, include_sale=False):
        self.m = m
        self.include_sale = include_sale
        super().__init__(N=N)

    def f(self, x, u, w, k):
        if u == "sale":
            return 0
        else:
            return max(0, min(2, x + u - w )) # the usual problem
    
    def g(self, x, u, w, k):
        if u == "sale":
            return 3/4*(self.m - w)
        else:
            return u + (x + u - w) ** 2

    def A(self, x, k):
        A = set(range(0,self.m-1+1))
        if self.include_sale: A.add("sale")
        return A

    def Pw(self, x, u, k):  # Distribution over random disturbances
        return {i: 1/self.m for i in range(0, self.m-1+1)}


def a_get_cost(N: int, m: int, x0 : int) -> float:
    model = BridalInventory(N=N, m=m, include_sale=False)
    J,pi = DP_stochastic(model)
    expected_cost = J[0][x0]
    return expected_cost


def b_sale(N : int, m : int, x0 : int) -> float:
    model = BridalInventory(N=N, m=m, include_sale=True)
    J,pi = DP_stochastic(model)
    expected_cost = J[0][x0]
    return expected_cost


if __name__ == "__main__":
    x0 = 0
    N = 6
    m = 4
    print(f"a) The expected cost should be 13.75, and you got {a_get_cost(N, m=m, x0=x0)=}")
    print(f"b) Expected cost when the sales-option is available should be approximately 11.25, and you got {b_sale(N, m=m, x0=x0)=}")
