'''
Random Walk

Author: Urinx
'''
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def RandomWalk(n=19):
    S = (n+1)//2
    t = []
    while S:
        S_ = S + np.random.choice([-1,1])
        if S_ == 20: R = 1
        elif S_ == 0: R = -1
        else: R = 0
        S_ %= n+1
        t.append((S, R, S_))
        S = S_
    return t

class n_step_TD:
    def __init__(self, n, α, γ=1):
        self.n = n
        self.α = α
        self.γ = γ
        self.V = [0] * 20
        self.V_ = np.linspace(-.9, .9, 19)

    def reset(self):
        self.V = [0] * 20

    def update(self, trajectory):
        V, n, α, γ = self.V, self.n, self.α, self.γ
        for i in range(len(trajectory)):
            S = trajectory[i][0]
            G = 0
            for j, (_, R, S_) in enumerate(trajectory[i:i+n]):
                G += γ**j * R
            G += γ**(j+1) * V[S_]
            V[S] += α * (G - V[S])

    def rms(self):
        return ((self.V_ - self.V[1:])**2).mean()**0.5

    def __str__(self):
        return f'[n-step TD] n: {self.n}, α: {self.α}, V: {self.V[1:]}'

def plot_result(ax, data):
    for i in range(len(data)):
        ax.plot(data[i][0], data[i][1], '-', label=f'n={2**i}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim((-0.02, 1))
    ax.set_ylim((0.24, 0.55))
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0.25, 0.55, 0.05))
    ax.set_xlabel(r'$\alpha$', fontsize=15)
    ax.set_ylabel('Average\nRMS error\nover 19 states\nand first 10\nepisodes', fontsize=14, rotation=0, labelpad=50, verticalalignment='center')
    ax.text(.92, .34, 'n=1', fontsize=14)
    ax.text(.61, .265, 'n=2', fontsize=14)
    ax.text(.35, .25, 'n=4', fontsize=14)
    ax.text(.18, .265, 'n=8', fontsize=14)
    ax.text(.04, .3, 'n=16', fontsize=14)
    ax.text(.37, .51, 'n=32', fontsize=14)
    ax.text(.2, .52, 'n=64', fontsize=14)
    ax.text(.12, .53, '128', fontsize=12)
    ax.text(.075, .551, '256', fontsize=10)
    ax.text(.03, .54, '512', fontsize=10)

def fig_7_2():
    runs = 100
    episodes = 10

    δ = 21
    data = [[np.linspace(0,  1,δ), np.zeros(δ)],
            [np.linspace(0,  1,δ), np.zeros(δ)],
            [np.linspace(0,  1,δ), np.zeros(δ)],
            [np.linspace(0,  1,δ), np.zeros(δ)],
            [np.linspace(0, .8,δ), np.zeros(δ)],
            [np.linspace(0, .5,δ), np.zeros(δ)],
            [np.linspace(0,.25,δ), np.zeros(δ)],
            [np.linspace(0,.15,δ), np.zeros(δ)],
            [np.linspace(0, .1,δ), np.zeros(δ)],
            [np.linspace(0, .1,δ), np.zeros(δ)]
    ]
    for r in tqdm(range(runs)):
        trajectories = [RandomWalk() for e in range(episodes)]

        for i in range(len(data)):
            for j in range(1, δ):
                α = data[i][0][j]
                agent = n_step_TD(2**i, α)
                for t in trajectories:
                    agent.update(t)
                    data[i][1][j] += agent.rms()
    else:
        for i in range(len(data)):
            data[i][1] /= episodes * runs
            data[i][1][0] = n_step_TD(1, 0).rms()

    plt.figure(figsize=(8, 6))
    ax = plt.subplot()
    plot_result(ax, data)
    plt.savefig(f'fig_7.2_random_walk.png', dpi=300, bbox_inches='tight')

# fig_7_2()