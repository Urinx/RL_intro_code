'''
Gridworld

Author: Urinx
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
from itertools import product

class Gridworld:
    def __init__(self, w=4, γ=1):
        self.w = w
        self.k = 0
        self.γ = γ
        self.V = np.zeros((w, w))

    def step(self, S, A):
        w = self.w
        S_, R = (S[0]+A[0], S[1]+A[1]), -1
        if S in [(0, 0), (w-1, w-1)]:  S_, R = S, 0
        elif -1 in S_ or self.w in S_: S_ = S
        return S_, R

    def policy_evaluation(self, in_place=True, hook=None):
        w, γ, Δ, θ, p = self.w, self.γ, 1, 1e-4, .25

        self.k = 0
        self.V = np.zeros((w, w))

        while Δ >= θ:
            Δ = 0
            if not in_place: V_ = np.zeros_like(self.V)

            for S in product(range(w), repeat=2):
                v = 0
                for A in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    S_, R = self.step(S, A)
                    v += p * (R + γ * self.V[S_])
                Δ = max(Δ, abs(v - self.V[S]))

                if in_place:
                    self.V[S] = v
                else:
                    V_[S] = v

            if not in_place: self.V = V_

            self.k += 1
            if hook is not None: hook(self.k, self.V)

    def plot_random_policy(self, out, in_place=True):
        w = self.w
        k_arr = [0, 1, 2, 3, 10, np.inf]
        Vk = {0: np.zeros((w, w))}
        
        def hook_func(k, v):
            if k in k_arr:
                Vk[k] = v.copy()

        self.policy_evaluation(in_place, hook_func)
        Vk[np.inf] = self.V.copy()
        print(f'in-place: {in_place}, k: {self.k}\n{self.V.round(1)}')

        plt.figure(figsize=(10, 35))
        for i in range(len(k_arr)):
            ax = plt.subplot(len(k_arr), 2, i*2+1)
            self.plot_gridvalue(ax, Vk[k_arr[i]].round(1))
            ax2 = plt.subplot(len(k_arr), 2, (i+1)*2)
            self.plot_gridpolicy(ax2, self.V2Q(Vk[k_arr[i]].round(1)))
            
            label = f'k={k_arr[i]}'
            if i == 0:
                ax.set_title('$v_k$ for the\nrandom policy', fontsize=30, pad=50)
                ax2.set_title('greedy policy\nw.r.t. $v_k$', fontsize=30, pad=50)
            elif i == len(k_arr)-1:
                label = r'k=$\infty$'

            ax.text(-.25, .5, label, fontsize=20)

        plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=1)
        plt.close()

    def V2Q(self, v):
        w = self.w
        Q = np.zeros((w, w, 4))
        for S in product(range(w), repeat=2):
            for i, A in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                S_, _ = self.step(S, A)
                Q[S][i] = v[S_]
        return Q

    def plot_gridvalue(self, ax, mat):
        ax.set_axis_off()
        tb = Table(ax, bbox=[0, 0, 1, 1])

        nrows, ncols = mat.shape
        width, height = 1.0 / ncols, 1.0 / nrows
        # Add cells
        for (i, j), val in np.ndenumerate(mat):
            tb.add_cell(i, j, width, height, text=val,
                        loc='center', facecolor='white')
        tb.set_fontsize(20)
        ax.add_table(tb)

    def plot_gridpolicy(self, ax, mat):
        ax.set_axis_off()
        tb = Table(ax, bbox=[0, 0, 1, 1])
        w = self.w
        d = 0.03
        width = height = 1.0 / w
        p2g = lambda p: (p[1]*width + width/2, 1 - p[0]*height - height/2)

        for i in range(w):
            for j in range(w):
                s = (i, j)
                if s in [(0, 0), (w-1, w-1)]:
                    tb.add_cell(i, j, width, height, text='', facecolor='gray')
                    continue
                else:
                    tb.add_cell(i, j, width, height, text='')

                m = mat[s].max()
                dire = mat[s] == m
                x, y = p2g(s)

                if all(dire == [1, 1, 1, 1]):
                    ax.arrow(x, y, 0, d, head_width=d, fc='k')
                    ax.arrow(x, y, 0, -d, head_width=d, fc='k')
                    ax.arrow(x, y, d, 0, head_width=d, fc='k')
                    ax.arrow(x, y, -d, 0, head_width=d, fc='k')
                elif all(dire == [1, 0, 0, 0]):
                    # up
                    ax.arrow(x, y - width / 2 + d, 0, width / 2, head_width=d, fc='k')
                elif all(dire == [0, 1, 0, 0]):
                    # down
                    ax.arrow(x, y + width / 2 - d, 0, - width / 2, head_width=d, fc='k')
                elif all(dire == [0, 0, 1, 0]):
                    # <-
                    ax.arrow(x + width / 2 - d, y, - width / 2, 0, head_width=d, fc='k')
                elif all(dire == [0, 0, 0, 1]):
                    # ->
                    ax.arrow(x - width / 2 + d, y, width / 2, 0, head_width=d, fc='k')
                elif all(dire == [1, 0, 0, 1]):
                    # ^
                    # |->
                    ax.arrow(x - width / 2 + 2*d, y - width / 2 + 2*d, 0, width / 2 - d, head_width=d, fc='k')
                    ax.arrow(x - width / 2 + 2*d, y - width / 2 + 2*d, width / 2 - d, 0, head_width=d, fc='k')
                elif all(dire == [1, 0, 1, 0]):
                    #   ^
                    # <-|
                    ax.arrow(x + width / 2 - 2*d, y - width / 2 + 2*d, 0, width / 2 - d, head_width=d, fc='k')
                    ax.arrow(x + width / 2 - 2*d, y - width / 2 + 2*d, - width / 2 + d, 0, head_width=d, fc='k')
                elif all(dire == [0, 1, 1, 0]):
                    # <-|
                    #   v
                    ax.arrow(x + width / 2 - 2*d, y + width / 2 - 2*d, - width / 2 + d, 0, head_width=d, fc='k')
                    ax.arrow(x + width / 2 - 2*d, y + width / 2 - 2*d, 0, - width / 2 + d, head_width=d, fc='k')
                elif all(dire == [0, 1, 0, 1]):
                    # |->
                    # v
                    ax.arrow(x - width / 2 + 2*d, y + width / 2 - 2*d, width / 2 - d, 0, head_width=d, fc='k')
                    ax.arrow(x - width / 2 + 2*d, y + width / 2 - 2*d, 0, - width / 2 + d, head_width=d, fc='k')
        ax.add_table(tb)

def fig_4_1():
    gw = Gridworld()
    # [The following comment is came from official source codes of the book]
    # While the author suggests using in-place iterative policy evaluation,
    # Figure 4.1 actually uses out-of-place version.
    gw.plot_random_policy('fig_4.1_policy_evaluation_on_gridworld_out-of-place.png', in_place=False)
    gw.plot_random_policy('fig_4.1_policy_evaluation_on_gridworld_in-place.png', in_place=True)

# fig_4_1()
