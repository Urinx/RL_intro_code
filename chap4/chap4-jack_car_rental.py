'''
Jack's Car Rental

Author: Urinx
'''
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import product
from functools import lru_cache

MOVE_CAR_COST = 2
RENT_CAR_CREDIT = 10
MAX_CARS = 20
POISSON_UPPER_BOUND = 11
λ_RENT_1 = 3
λ_RENT_2 = 4
λ_RETURN_1 = 3
λ_RETURN_2 = 2
MAX_MOVE_CAR = 5

@lru_cache(None)
def Poisson(n, λ):
    return (λ**n / np.math.factorial(n)) * np.e ** (-λ)

class CarRental:
    def __init__(self, discount=0.9, constant_ret=True):
        self.γ = discount
        self.π = np.zeros((MAX_CARS+1, MAX_CARS+1), dtype=int)
        self.V = np.zeros((MAX_CARS+1, MAX_CARS+1))
        self.k = 0
        self.constant_ret = constant_ret

    def step(self, S, A):
        s_day = (min(S[0]-A, MAX_CARS), min(S[1]+A, MAX_CARS))
        cost = - MOVE_CAR_COST * abs(A)

        for rent1, rent2 in product(range(POISSON_UPPER_BOUND), repeat=2):
            p_rent = Poisson(rent1, λ_RENT_1) * Poisson(rent2, λ_RENT_2)

            valid_rent1 = min(rent1, s_day[0])
            valid_rent2 = min(rent2, s_day[1])
            credit = (valid_rent1 + valid_rent2) * RENT_CAR_CREDIT
            s_left = (s_day[0] - valid_rent1, s_day[1] - valid_rent2)

            if self.constant_ret:
                ret1 = λ_RETURN_1
                ret2 = λ_RETURN_2
                p = p_rent
                S_ = (min(s_left[0]+ret1, MAX_CARS), min(s_left[1]+ret2, MAX_CARS))
                R = cost + credit
                yield S_, R, p
            else:
                for ret1, ret2 in product(range(POISSON_UPPER_BOUND), repeat=2):
                    p_ret = Poisson(ret1, λ_RETURN_1) * Poisson(ret2, λ_RETURN_2)
                    p = p_rent * p_ret
                    S_ = (min(s_left[0]+ret1, MAX_CARS), min(s_left[1]+ret2, MAX_CARS))
                    R = cost + credit
                    yield S_, R, p

    def policy_iteration(self, hook=None):
        n = MAX_CARS + 1
        γ, θ, V, π = self.γ, 1e-4, self.V, self.π
        action_space = np.arange(-MAX_MOVE_CAR, MAX_MOVE_CAR+1)

        while 1:
            self.k += 1
            # policy evaluation (in-place)
            while 1:
                Δ = 0
                for S in product(range(n), repeat=2):
                    A = self.π[S]
                    v = sum(p * (R + γ * V[S_]) for S_, R, p in self.step(S, A))
                    Δ = max(Δ, abs(v - V[S]))
                    V[S] = v
                if Δ < θ: break

            # policy improvement
            policy_stable = True
            for S in product(range(n), repeat=2):
                Q = []
                for A in action_space:
                    if (0 <= A <= S[0]) or (-S[1] <= A <= 0):
                        q = sum(p * (R + γ * V[S_]) for S_, R, p in self.step(S, A))
                        Q.append((q, A))
                if π[S] != (act := max(Q)[1]):
                    π[S] = act
                    policy_stable = False

            if hook is not None: hook(self.k, self.π)
            if policy_stable: break

def step_contour(ax, mat):
    for k in range(int(mat.min()), int(mat.max())+1):
        x = []
        y = []
        for j in range(mat.shape[1]):
            is_find = False
            for i in range(mat.shape[0]):
                if mat[i, j] == k:
                    y.append(i)
                    is_find = True
                    break
            if is_find:
                x.append(j)
        if k <= 0:
            for i in range(len(y)):
                if y[i] != 0: break
            x = x[i-1:]
            y = y[i-1:]
        else:
            x.append(x[-1]+1)
            y.append(21)
        ax.step(x, y, 'k', linewidth=1)
        # ax.plot(x, y, f'C{3+k}o', alpha=0.5)
        ax.text(x[len(x)//2]-0.8, y[len(y)//2]+.2, k, fontsize=20)
    ax.set_xlim((0, 20))
    ax.set_ylim((0, 21))
    [s.set_linewidth(1.5) for s in ax.spines.values()]

def fig_4_2(plot_heatmap=False):
    CONSTANT_RET = True
    car_rental = CarRental(constant_ret=CONSTANT_RET)
    
    arr = [np.zeros((MAX_CARS+1, MAX_CARS+1), dtype=int)]
    def foo(k, π):
        if k in [1, 2, 3, 4]: arr.append(π.copy())

    start_t = datetime.now()
    car_rental.policy_iteration(foo)
    end_t = datetime.now()

    print(f'k={car_rental.k}, time: {end_t-start_t}')
    print(f'π:\n{np.flipud(car_rental.π)}')
    arr.append(car_rental.V.round(0))

    if plot_heatmap: figsize = (40, 20)
    else: figsize = (40, 28)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()
    for i in range(len(arr)):
        ax = axes[i]

        if plot_heatmap:
            import seaborn as sns
            sns.heatmap(np.flipud(arr[i]), cmap="YlGnBu", ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            if i == len(arr)-1:
                # the 3d image show in book is plotted by mathematical,
                # here try my best to plot it as same as the original one.
                from mpl_toolkits.mplot3d import Axes3D
                ax = plt.subplot(236, projection='3d')
                x = y = np.arange(21)
                X, Y = np.meshgrid(x, y)
                ax.plot_wireframe(X, Y, arr[i][X,Y], lw=1, rstride=1, cstride=1, color='k')
                # make the panes transparent
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                # make the grid lines transparent
                ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
                ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
                ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
                ax.set_xlabel('#Cars at second location', fontsize=15)
                ax.set_ylabel('#Cars at first location', fontsize=15)
                ax.set_xlim((0, 20))
                ax.set_ylim((0, 20))
                ax.set_zlim((420, 612))
                ax.set_xticks([0, 20])
                ax.set_yticks([0, 20])
                ax.set_zticks([420, 612])
            else:
                step_contour(ax, arr[i])
                ax.set_xticks([])
                ax.set_yticks([])

        if i == len(arr)-1: ax.set_title(f'$v_k$', fontsize=40, pad=40)
        else: ax.set_title(f'$\pi_{i}$', fontsize=40, pad=40)

    plt.savefig(f'fig_4.2_jacks_car_rental{"_constant_return" if CONSTANT_RET else ""}.png', dpi=300, bbox_inches='tight')
    plt.close()

# fig_4_2()
