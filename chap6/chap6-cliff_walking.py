'''
Cliff Walking

Author: Urinx
'''
import numpy as np
from tqdm import tqdm
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.table import Table
import matplotlib.patches as patches
from concurrent.futures import ProcessPoolExecutor

class Act:
    U = (1,0)
    D = (-1, 0)
    R = (0, 1)
    L = (0, -1)
    N = 4

class CliffWorld:
    START = (0, 0)
    GOAL = (0, 11)
    H = 4
    W = 12
    Acts = [Act.U, Act.D, Act.R, Act.L]

    def __init__(self):
        self.reset()

    def reset(self):
        self.episodes = []
        self.reword = 0

    def step(self, S, A):
        A = self.Acts[A]
        S_ = (
            np.clip(S[0]+A[0], 0, self.H-1),
            np.clip(S[1]+A[1], 0, self.W-1)
        )

        if S_[0] == 0 and 0 < S_[1] < self.W-1:
            R = -100
            S_ = self.START
        else:
            R = -1

        self.reword += R
        if S_ == self.GOAL:
            self.episodes.append(self.reword)
            self.reword = 0
        return S_, R

class TD:
    def __init__(self, α=0.5, ε=0.1, γ=1):
        self.α = α
        self.ε = ε
        self.γ = γ
        self.reset()

    def reset(self):
        self.Q = {}

    def act(self, S):
        if np.random.rand() >= self.ε and S in self.Q:
            A = np.argmax(self.Q[S])
        else:
            A = np.random.randint(Act.N)
        return A

    def update(self, S, A, R, S_, A_=None):
        raise NotImplemented

class Sarsa(TD):
    def __init__(self, α=0.5):
        super().__init__()
        self.α = α

    def update(self, S, A, R, S_, A_):
        Q, α, γ = self.Q, self.α, self.γ
        if S not in Q:
            Q[S] = [0] * Act.N
        if S_ not in Q:
            Q[S_] = [0] * Act.N
        Q[S][A] += α * (R + γ * Q[S_][A_] - Q[S][A])

class ExpectedSarsa(TD):
    def __init__(self, α=0.5):
        super().__init__()
        self.α = α

    def update(self, S, A, R, S_):
        Q, α, ε, γ = self.Q, self.α, self.ε, self.γ
        π = np.ones(Act.N) / Act.N
        if S not in Q:
            Q[S] = [0] * Act.N
        if S_ not in Q:
            Q[S_] = [0] * Act.N
        else:
            π *= ε
            π[np.argmax(Q[S_])] += 1 - ε
        Q[S][A] += α * (R + γ * sum((π * Q[S_])) - Q[S][A])

class QLearning(TD):
    def __init__(self, α=0.5):
        super().__init__()
        self.α = α

    def update(self, S, A, R, S_):
        Q, α, γ = self.Q, self.α, self.γ
        if S not in Q:
            Q[S] = [0] * Act.N
        if S_ not in Q:
            Q[S_] = [0] * Act.N
        Q[S][A] += α * (R + γ * max(Q[S_]) - Q[S][A])

def plot_cliff_world(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    H = 0.5
    row, col = CliffWorld.H, CliffWorld.W
    Δh, Δw = 1/row, 1/col

    tb = Table(ax, bbox=[0, H, 1, 1-H])
    for i in range(row-1): tb.add_cell(i, 0, Δw, Δh, text='').set_lw(.5)
    tb.add_cell(row-1, 11, Δw, Δh, text='').set_lw(.5)
    c = tb.add_cell(row-1, 1, Δw*10, Δh, text='The Cliff', loc='center')
    c.set_lw(.5)
    c.set_color('#aaaaaa')

    tb2 = Table(ax, bbox=[0, H, 1, 1-H])
    for i in range(row-1):
        for j in range(col):
            tb2.add_cell(i, j, Δw, Δh, text='').set_lw(.5)
    i = row-1
    tb2.add_cell(i, 0, Δw, Δh, text='S', loc='center').set_lw(.5)
    tb2.add_cell(i, 11, Δw, Δh, text='G', loc='center').set_lw(.5)

    ax.add_table(tb)
    ax.add_table(tb2)

    ax.set_title(r'$R=-1$')
    ax.text(-0.01, H + (1-H)/row*3.5, 'safe path', fontsize=12, horizontalalignment='right', verticalalignment='center')
    ax.text(-0.01, H + (1-H)/row*.5, 'optimal path', fontsize=12, horizontalalignment='right', verticalalignment='center')

    p2p = lambda p: (Δw*(p+.5), H)
    for i, p in enumerate([1, 2, 5, 10]):
        arrow = patches.FancyArrowPatch(p2p(p), (Δw/5*(4-i), H), connectionstyle="angle3, angleA=45,angleB=-70",
                arrowstyle="Simple,tail_width=0.5,head_width=4,head_length=6", color="k")
        ax.add_artist(arrow)

    ax.text(0.3, H-0.05, '•   •   •', fontsize=13, horizontalalignment='center', verticalalignment='center')
    ax.text(0.64, H-0.05, '•     •     •', fontsize=13, horizontalalignment='center', verticalalignment='center')
    ax.text(0.5, H-0.2, r'R=-100', fontsize=13, horizontalalignment='center', verticalalignment='center')

    ps = [
        [(Δw*.5, H + (1-H)/row * 0.8), (Δw*.5, H + (1-H)/row * 3.5), .04],
        [(Δw*.6, H + (1-H)/row * 3.5), (1-Δw*.6, H + (1-H)/row * 3.5), .03],
        [(1-Δw*.5, H + (1-H)/row * 3.5), (1-Δw*.5, H + (1-H)/row * 0.8), .04],
        [(Δw*.6, H + (1-H)/row * 1.5), (1-Δw*.6, H + (1-H)/row * 1.5), .03]
    ]
    for p1, p2, d in ps:
        ax.arrow(*p1, p2[0]-p1[0], p2[1]-p1[1], head_width=.012, head_length=d, length_includes_head=True, color='#9e9e9e')

def plot_sarsa_q_result(ax, data):
    ax.plot(data[1], '-', lw=1, color='gray')
    ax.plot(data[0], 'k-', lw=1)
    ax.text(250, -21, 'Sarsa', fontsize=13, horizontalalignment='center', verticalalignment='center')
    ax.text(250, -60, 'Q-learning', color='gray', fontsize=13, horizontalalignment='center', verticalalignment='center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_bounds(0, 500)
    ax.set_xlim((0, 500))
    ax.set_ylim((-100, -20))
    ax.set_yticks([-100, -75, -50, -25])
    ax.set_xlabel('Episodes', fontsize=13)
    ax.set_ylabel('Sum of\nrewards\nduring\nepisode', fontsize=13, rotation=0, labelpad=30)

def plot_performance(ax, interim_data, asymptotic_data):
    αs = np.linspace(0.1, 1, 19)
    ax.plot(αs, interim_data[0], 'v:', fillstyle='none', color='tab:blue')
    ax.plot(αs, interim_data[1], 'x:', fillstyle='none', color='tab:red')
    ax.plot(αs, interim_data[2], 's:', fillstyle='none', color='k')
    ax.plot(αs, asymptotic_data[0], 'v-', fillstyle='none', label='Sarsa', color='tab:blue')
    ax.plot(αs, asymptotic_data[1], 'x-', fillstyle='none', label='Expexted Sarsa', color='tab:red')
    ax.plot(αs, asymptotic_data[2], 's-', fillstyle='none', label='Q-learning', color='k')
    ax.set_xlim((0.1, 1))
    ax.set_ylim((-160, 0))
    ax.set_yticks([-120, -80, -40, 0])
    ax.set_xlabel(r'$\alpha$', fontsize=16)
    ax.set_ylabel('Sum of rewards\nper episode', fontsize=13, rotation=0, labelpad=40)
    ax.text(0.2, -40, 'Asymptotic Performance', fontsize=13, color='tab:green')
    ax.text(0.3, -110, 'Interim Performance', fontsize=13, color='tab:green')
    ax.legend()

def one_run(agent, world, episodes):
    world.reset()
    agent.reset()
    while len(world.episodes) < episodes:
        S = world.START
        A = agent.act(S)

        while S != world.GOAL:
            S_, R = world.step(S, A)
            if type(agent) is Sarsa:
                A_ = agent.act(S_)
                agent.update(S, A, R, S_, A_)
            else:
                agent.update(S, A, R, S_)
                A_ = agent.act(S_)
            S, A = S_, A_
    return world.episodes

def exp_6_6():
    runs = 1000
    episodes = 500
    agents = [Sarsa(), QLearning()]
    world = CliffWorld()

    data = np.zeros((len(agents), episodes))
    for r in tqdm(range(runs)):
        for i, agent in enumerate(agents):
            data[i] += one_run(agent, world, episodes)
    else:
        data /= runs

    fig, axs = plt.subplots(2, 1, figsize=(6, 9))
    plot_cliff_world(axs[0])
    plot_sarsa_q_result(axs[1], data)
    plt.savefig(f'exp_6.6_cliff_walking.png', dpi=300, bbox_inches='tight')

METHODS = [Sarsa, ExpectedSarsa, QLearning]

def one_run_proc(p):
    return np.mean(one_run(METHODS[p[0]](0.1 + 0.05 * p[1]), CliffWorld(), p[2]))

def fig_6_3():
    asymptotic_episodes = 100000
    asymptotic_runs = 10
    interim_episodes = 100
    interim_runs = 50000
    num_of_workers = 20

    def test_α_mp(runs, episodes, workers=4):
        data = np.zeros((3, 19))
        tasks = list(product(range(3), range(19), [episodes]))

        with ProcessPoolExecutor(max_workers=workers) as executor:
            for i in tqdm(range(runs)):
                for idx, r in zip(tasks, executor.map(one_run_proc, tasks)):
                    data[idx[:2]] += r
            else:
                data /= runs
        return data

    interim_data = test_α_mp(interim_runs, interim_episodes, num_of_workers)
    asymptotic_data = test_α_mp(asymptotic_runs, asymptotic_episodes, num_of_workers)

    plt.figure(figsize=(7, 6))
    ax = plt.subplot()
    plot_performance(ax, interim_data, asymptotic_data)
    plt.savefig(f'fig_6.3_cliff_walking.png', dpi=300, bbox_inches='tight')

# exp_6_6()
# fig_6_3()
