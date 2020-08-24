'''
Gridworld

Author: Urinx
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.table import Table
from itertools import product

class Gridworld:
    def __init__(self, w=5, A=(0, 1), B=(0, 3), A_=(4, 1), B_=(2, 3), γ=.9):
        self.w = w
        self.A = A
        self.B = B
        self.A_ = A_
        self.B_ = B_
        self.γ = γ
        self.V = np.zeros((w, w))
        self.V_n = 0
        self.V_opt = np.zeros((w, w))
        self.Q_opt = np.zeros((w, w, 4))

    def step(self, S, A):
        S_, R = (S[0]+A[0], S[1]+A[1]), 0
        if S == self.A: S_, R = self.A_, 10
        elif S == self.B: S_, R = self.B_, 5
        elif -1 in S_ or self.w in S_: S_, R = S, -1
        return S_, R

    def estimate_V(self):
        p, w, γ, V = .25, self.w, self.γ, self.V
        self.V_n = 0

        while 1:
            V_ = np.zeros((w, w))
            for S in product(range(w), repeat=2):
                for A in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    S_, R = self.step(S, A)
                    V_[S] += p * (R + γ * self.V[S_])
            if abs(V_ - self.V).sum() < 1e-4: break
            self.V = V_
            self.V_n += 1

    def estimate_V_and_Q_opt(self):
        w, γ = self.w, self.γ
        self.V_n = 0

        while 1:
            Q = np.zeros((w, w, 4))
            for S in product(range(w), repeat=2):
                for i, A in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                    S_, R = self.step(S, A)
                    Q[S][i] = R + γ * self.Q_opt[S_].max()
            if abs(Q - self.Q_opt).sum() < 1e-4: break
            self.Q_opt = Q
            self.V_n += 1
        self.V_opt = self.Q_opt.max(-1)

    def plot_gridworld(self, ax):
        fs = 20
        ax.set_axis_off()
        tb = Table(ax, bbox=[0, 0, 1, 1])
        w = self.w
        width = height = 1.0 / w
        # Add cells
        for i in range(w):
            for j in range(w):
                val = ''
                if (i, j) == self.A:
                    val = 'A'
                elif (i, j) == self.B:
                    val = 'B'
                elif (i, j) == self.A_:
                    val = 'A\''
                elif (i, j) == self.B_:
                    val = 'B\''
                tb.add_cell(i, j, width, height, text=val,
                        loc='center', facecolor='white')
        tb.set_fontsize(fs)
        ax.add_table(tb)

        p2g = lambda p: (p[1]*width + width/2 + 0.05, 1 - p[0]*height - height/2)
        for x, y, t in [(p2g(self.A), p2g(self.A_), '+10'), (p2g(self.B), p2g(self.B_), '+5')]:
            arrow = patches.FancyArrowPatch(x, y, connectionstyle="arc3,rad=-.5",
                arrowstyle="Simple,tail_width=0.5,head_width=4,head_length=8", color="k")
            ax.add_artist(arrow)
            plt.text((x[0]+y[0])/2, (x[1]+y[1])/2, t, fontsize=fs-2)

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

    def plot_gridpolicy(self, ax):
        ax.set_axis_off()
        tb = Table(ax, bbox=[0, 0, 1, 1])
        w = self.w
        d = 0.03
        width = height = 1.0 / w
        p2g = lambda p: (p[1]*width + width/2, 1 - p[0]*height - height/2)

        for i in range(w):
            for j in range(w):
                tb.add_cell(i, j, width, height, text='')
                s = (i, j)
                m = self.Q_opt[s].max()
                dire = self.Q_opt[s] == m
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
                    ax.arrow(x - width / 2 + 2*d, y - width / 2 + 2*d, 0, width / 2 - d, head_width=d, fc='k')
                    ax.arrow(x - width / 2 + 2*d, y - width / 2 + 2*d, width / 2 - d, 0, head_width=d, fc='k')
                elif all(dire == [1, 0, 1, 0]):
                    ax.arrow(x + width / 2 - 2*d, y - width / 2 + 2*d, 0, width / 2 - d, head_width=d, fc='k')
                    ax.arrow(x + width / 2 - 2*d, y - width / 2 + 2*d, - width / 2 + d, 0, head_width=d, fc='k')
                elif all(dire == [0, 1, 1, 0]):
                    ax.arrow(x + width / 2 - 2*d, y + width / 2 - 2*d, - width / 2 + d, 0, head_width=d, fc='k')
                    ax.arrow(x + width / 2 - 2*d, y + width / 2 - 2*d, 0, - width / 2 + d, head_width=d, fc='k')
                elif all(dire == [0, 1, 0, 1]):
                    ax.arrow(x - width / 2 + 2*d, y + width / 2 - 2*d, width / 2 - d, 0, head_width=d, fc='k')
                    ax.arrow(x - width / 2 + 2*d, y + width / 2 - 2*d, 0, - width / 2 + d, head_width=d, fc='k')
        ax.add_table(tb)

def fig_3_2():
    gw = Gridworld()
    gw.estimate_V()
    print(f'Iteration times: {gw.V_n}')
    print(f'Value function:\n{np.round(gw.V, 1)}')
    
    plt.figure(figsize=(11, 5))
    gw.plot_gridworld(plt.subplot(121))
    gw.plot_gridvalue(plt.subplot(122), np.round(gw.V, 1))
    plt.savefig('fig_3.2_gridworld_v.png', dpi=300)
    plt.close()

def fig_3_5():
    gw = Gridworld()
    gw.estimate_V_and_Q_opt()
    print(f'Iteration times: {gw.V_n}')
    print(f'Optimal Value function:\n{np.round(gw.V_opt, 1)}')

    plt.figure(figsize=(17, 5))
    gw.plot_gridworld(plt.subplot(131))
    gw.plot_gridvalue(plt.subplot(132), np.round(gw.V_opt, 1))
    gw.plot_gridpolicy(plt.subplot(133))
    plt.savefig('fig_3.5_gridworld_optimal_solutions.png')
    plt.close()

# fig_3_2()
# fig_3_5()
