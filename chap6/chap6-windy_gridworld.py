'''
Windy Gridworld

Author: Urinx
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
import matplotlib.patches as patches

class Act:
    U = (1,0)
    D = (-1, 0)
    R = (0, 1)
    L = (0, -1)
    N = 4

class WindyGridworld:
    START = (3, 0)
    GOAL = (3, 7)
    WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    H = 7
    W = 10
    Acts = [Act.U, Act.D, Act.R, Act.L]

    def __init__(self):
        self.episodes = [0]
        self.steps = 0

    def step(self, S:tuple, A:int) -> (tuple, int):
        A = self.Acts[A]
        S_ = (
            np.clip(S[0]+A[0]+self.WIND[S[1]], 0, self.H-1),
            np.clip(S[1]+A[1], 0, self.W-1)
        )
        self.steps += 1
        if S_ == self.GOAL:
            self.episodes.append(self.steps)
        return S_, -1

class Sarsa:
    def __init__(self, α=0.5, ε=0.1, γ=1):
        self.α = α
        self.ε = ε
        self.γ = γ
        self.Q = {}

    def act(self, S:tuple) -> Act:
        if np.random.rand() >= self.ε and S in self.Q:
            A = np.argmax(self.Q[S])
        else:
            A = np.random.randint(Act.N)
        return A

    def update(self, S, A, R, S_, A_):
        Q, α, ε, γ = self.Q, self.α, self.ε, self.γ
        if S not in Q:
            Q[S] = [0] * Act.N
        if S_ not in Q:
            Q[S_] = [0] * Act.N
        Q[S][A] += α * (R + γ * Q[S_][A_] - Q[S][A])

def plot_episodes_result(ax, data, trajectory):
    ax.plot(data, range(len(data)), 'k-', lw=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_bounds(-5, 170)
    ax.set_xlim((-150, 8000))
    ax.set_ylim((-5, 180))
    ax.set_yticks([0, 50, 100, 150, 170])
    ax.set_xlabel('Time steps', fontsize=15)
    ax.set_ylabel('Episodes', fontsize=15, labelpad=10)

    center = (5200, 132)
    d = 10
    ax.arrow(*center, 0, d, head_width=50, head_length=.3*d, length_includes_head=True, fc='k')
    ax.arrow(*center, 0, -d, head_width=50, head_length=.3*d, length_includes_head=True, fc='k')
    ax.arrow(*center, d*30, 0, head_width=d/6, head_length=.3*d*30, length_includes_head=True, fc='k')
    ax.arrow(*center, -d*30, 0, head_width=d/6, head_length=.3*d*30, length_includes_head=True, fc='k')
    ax.text(center[0], center[1]-2*d, 'Actions', fontsize=10, horizontalalignment='center')

    tb = Table(ax, bbox=[0.09, 0.42, 0.45, 0.57])
    tb.set_fontsize(12)
    row, col = 8, 10
    h, w = 1/row, 1/col
    for i in range(row):
        for j in range(col):
            if (i,j) == WindyGridworld.START: t = 'S'
            elif (i,j) == WindyGridworld.GOAL: t = 'G'
            elif i == 7: t = WindyGridworld.WIND[j]
            else: t = ''
            c = tb.add_cell(i, j, w, h, text=t, loc='center')
            if i == 7:
                c.set_lw(0)
            else:
                c.set_ec('gray')
                c.set_lw(.5)

    ax.add_table(tb)
    ax.add_patch(
        patches.Rectangle((590, 86), 3660, 92, fill=False)
    )
    ax.arrow(2800, 90, 0, 30, width=200, head_width=400, head_length=10, length_includes_head=True, fc='gray', ec='gray')
    ax.arrow(2800, 130, 0, 30, width=200, head_width=400, head_length=10, length_includes_head=True, fc='gray', ec='gray')

    dh, dw = 13, 365
    for i in range(1, len(trajectory)):
        s, s_ = trajectory[i-1], trajectory[i]
        if i == 1:
            dx = 100 if s_[1] > s[1] else 0
            dy = 5 if s_[0] > s[0] else 0
            p = (780+dx, 132+dy)
        elif i == len(trajectory)-1:
            dx = -110 if s_[1] < s[1] else 0
            dy = 4 if s_[0] > s[0] else 0
        else:
            dx = dy = 0
        q = (p[0]+dw*(s_[1]-s[1]) - dx, p[1]+dh*(s_[0]-s[0]) - dy)
        ax.plot([p[0], q[0]], [p[1], q[1]], 'k-')
        p = q

def exp_6_5():
    steps = 8000
    agent = Sarsa()
    world = WindyGridworld()

    S = world.START
    A = agent.act(S)
    for _ in range(5*steps):
        S_, R = world.step(S, A)
        A_ = agent.act(S_)
        agent.update(S, A, R, S_, A_)
        if S_ == world.GOAL:
            S = world.START
            A = agent.act(S)
        else:
            S, A = S_, A_

    S = world.START
    trajectory = [S]
    while S != world.GOAL:
        A = agent.act(S)
        S_, _ = world.step(S, A)
        if S_ != trajectory[-1]:
            trajectory.append(S_)
        S = S_

    plt.figure(figsize=(8, 5))
    ax = plt.subplot()
    plot_episodes_result(ax, world.episodes, trajectory)
    plt.savefig('exp_6.5_windy_gridworld.png', dpi=300, bbox_inches='tight')

# exp_6_5()
