'''
Maximization Bias

Author: Urinx
'''
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MDP:
    START = 'A'
    MID = 'B'
    END = 'T'
    L = 1
    START_ACT_NUM = 2
    MID_ACT_NUM = 10

    @staticmethod
    def step(S, A):
        return (MDP.MID if A == MDP.L else MDP.END, 0) if S == MDP.START else (MDP.END, np.random.normal(-.1, 1))

class TD:
    def __init__(self, α=0.1, ε=0.1, γ=1, double=False):
        self.α = α
        self.ε = ε
        self.γ = γ
        self.double = double
        self.reset()

    def reset(self):
        self.Q = self.init_Q()
        if self.double: self.Q_ = self.init_Q()

    def init_Q(self):
        return {MDP.START: np.zeros(MDP.START_ACT_NUM), MDP.MID: np.zeros(MDP.MID_ACT_NUM), MDP.END: np.zeros(1)}

    def act(self, S):
        if np.random.rand() >= self.ε:
            V = (self.Q[S]+self.Q_[S]) if self.double else self.Q[S]
            A = np.random.choice(np.where(V==V.max())[0])
        else:
            A = np.random.randint(MDP.START_ACT_NUM if S == MDP.START else MDP.MID_ACT_NUM)
        return A

    def update(self, S, A, R, S_, A_=None):
        raise NotImplemented

class QLearning(TD):
    def update(self, S, A, R, S_):
        Q, α, γ = self.Q, self.α, self.γ
        Q[S][A] += α * (R + γ * max(Q[S_]) - Q[S][A])

class DoubleQLearning(TD):
    def __init__(slef):
        super().__init__(double=True)

    def update(self, S, A, R, S_):
        α, γ = self.α, self.γ
        if np.random.binomial(1, 0.5) == 1:
            Q, Q_ = self.Q, self.Q_
        else:
            Q, Q_ = self.Q_, self.Q
        A_ = np.argmax(Q[S_])
        Q[S][A] += α * (R + γ * Q_[S_][A_]- Q[S][A])

def plot_result(ax, data):
    ax.plot(np.arange(1,data.shape[-1]+1), data[0], '-', color='tab:red', lw=1)
    ax.text(120, 0.45, 'Q-learning', color='tab:red', fontsize=14)
    ax.plot(np.arange(1,data.shape[-1]+1), data[1], '-', color='tab:green', lw=1)
    ax.text(40, 0.25, 'Double\nQ-learning', color='tab:green', fontsize=14)
    ax.hlines(0.05, -3, 300, linestyles='--', color='k', lw=1)
    ax.text(300, 0.04, 'optimal', fontsize=12, horizontalalignment='right', verticalalignment='top')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim((-3, 300))
    ax.set_ylim((-0.02, 1))
    ax.set_xticks([1, 100, 200, 300])
    ax.set_yticks([0, 0.05, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels([0, '5%', '25%', '50%', '75%', '100%'])
    ax.set_xlabel('Episodes', fontsize=14)
    ax.set_ylabel('% left\nactions\nfrom A', fontsize=14, rotation=0, labelpad=30)

    Δ = 30
    ax.add_patch(patches.Rectangle((Δ+100, .8), 10, .05, fc='#aaaaaa', ec='k'))
    ax.plot(Δ+150, .825, 'ko', markersize=25, fillstyle='none')
    ax.text(Δ+150, .825, 'B', fontsize=14, horizontalalignment='center', verticalalignment='center')
    arrowstyle="Simple,tail_width=0.5,head_width=4,head_length=6"
    ax.add_artist(patches.FancyArrowPatch((Δ+145, .85), (Δ+110, .85), connectionstyle="arc3, rad=.7", arrowstyle=arrowstyle, color="k"))
    ax.add_artist(patches.FancyArrowPatch((Δ+143, .83), (Δ+110, .83), connectionstyle="arc3, rad=.2", arrowstyle=arrowstyle, color="k"))
    ax.add_artist(patches.FancyArrowPatch((Δ+145, .8), (Δ+110, .8), connectionstyle="arc3, rad=-.7", arrowstyle=arrowstyle, color="k"))
    ax.plot([Δ+127]*3, [.74, .845, .91], 'ko', markersize=4)
    ax.plot([Δ+127]*3, [.766, .792, .818], 'o', markersize=1, color='gray')
    ax.text(Δ+127, .92, r'$\mathcal{N}(-0.1,1)$', fontsize=12, horizontalalignment='center', verticalalignment='bottom')
    ax.plot(Δ+195, .825, 'ko', markersize=25, fillstyle='none')
    ax.text(Δ+195, .825, 'A', fontsize=14, horizontalalignment='center', verticalalignment='center')
    ax.arrow(Δ+204, .825, 30, 0, head_width=.01, head_length=4, length_includes_head=True, color='k')
    ax.plot(Δ+219, .825, 'ko', markersize=4)
    ax.text(Δ+219, .835, '0', fontsize=10, horizontalalignment='center', verticalalignment='bottom')
    ax.text(Δ+219, .81, 'right', fontsize=10, horizontalalignment='center', verticalalignment='top')
    ax.arrow(Δ+186, .825, -28, 0, head_width=.01, head_length=4, length_includes_head=True, color='k')
    ax.plot(Δ+172, .825, 'ko', markersize=4)
    ax.text(Δ+172, .835, '0', fontsize=10, horizontalalignment='center', verticalalignment='bottom')
    ax.text(Δ+172, .81, 'left', fontsize=10, horizontalalignment='center', verticalalignment='top')
    ax.add_patch(patches.Rectangle((Δ+235, .8), 10, .05, fc='#aaaaaa', ec='k'))

def fig_6_5():
    runs = 10000
    episodes = 300
    agents = [QLearning(), DoubleQLearning()]

    left_count = np.zeros((len(agents), episodes))
    for r in tqdm(range(runs)):
        for i, agent in enumerate(agents):
            agent.reset()

            for e in range(episodes):
                S = MDP.START
                while S != MDP.END:
                    A = agent.act(S)
                    if S == MDP.START: left_count[i][e] += A
                    S_, R = MDP.step(S, A)
                    agent.update(S, A, R, S_)
                    S = S_
    else:
        left_count /= runs

    plt.figure(figsize=(9, 6))
    ax = plt.subplot()
    plot_result(ax, left_count)
    plt.savefig(f'fig_6.5_maximization_bias.png', dpi=300, bbox_inches='tight')

fig_6_5()