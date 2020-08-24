'''
Random Walk (MRPs)

Author: Urinx
'''
import queue
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def RandomWalk():
    S = 3
    while S:
        S_ = S + np.random.choice([-1,1])
        R = 1 if S_ == 6 else 0
        S_ %= 6
        yield S, R, S_
        S = S_

class TD:
    def __init__(self, α=0.1, γ=1):
        self.α = α
        self.γ = γ
        self.V = [0] + [0.5] * 5
        self.V_ = np.arange(1,6) / 6
        self.deque = queue.deque()

    def update(self, S, R, S_):
        V, α, γ = self.V, self.α, self.γ
        V[S] += α * (R + γ * V[S_] - V[S])
        self.deque.append((S, R, S_))

    def update_batch(self):
        V, V_, α, γ, θ = self.V, np.array(self.V[1:]), self.α, self.γ, 1e-3
        while 1:
            for S, R, S_ in self.deque:
                V[S] += α * (R + γ * V[S_] - V[S])
            if abs(V_ - V[1:]).sum() < θ: break
            V_ = np.array(V[1:])

    def rms(self):
        return ((self.V_ - self.V[1:])**2).mean()**0.5

    def __str__(self):
        return f'[TD] α: {self.α}, V: {self.V[1:]}'

class MC:
    # every-visit constant-α Monte Carlo
    def __init__(self, α=0.1):
        self.α = α
        self.trajectory = []
        self.V = [0.5] * 5
        self.V_ = np.arange(1,6) / 6
        self.deque = queue.deque()

    def update(self, S, R, S_):
        self.trajectory.append(S-1)

        if S_ == 0:
            α, V, G = self.α, self.V, R
            for s in self.trajectory:
                V[s] = V[s] + α * (G - V[s])
                self.deque.append((s, G))
            self.trajectory = []

    def update_batch(self):
        α, V, V_, θ = self.α, self.V, np.array(self.V), 1e-3
        while 1:
            for S, G in self.deque:
                V[S] = V[S] + α * (G - V[S])
            if abs(V_ - V).sum() < θ: break
            V_ = np.array(V)

    def rms(self):
        return ((self.V_ - self.V)**2).mean()**0.5

    def __str__(self):
        return f'[MC] α: {self.α}'

def plot_estimated_value(ax, data):
    for d in data:
        e = d['e']
        label = 'pred values' if e==100 else None
        lw = 1 if e >= 10 else 0.8
        ax.plot(d['v'], 'o-', color='black', lw=lw, ms=2, label=label)
        ax.text(-0.05, d['v'][0], e, horizontalalignment='right', verticalalignment='center')
    ax.plot([i/6 for i in range(1,6)], 'o--', color='black', linewidth=0.8, ms=2, label='true values')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_bounds(-0.4, 4)
    ax.set_xlim((-0.4, 4.1))
    ax.set_xticks(range(5))
    ax.set_xticklabels(['A', 'B', 'C', 'D', 'E'])
    ax.set_xlabel('State', fontsize=15)
    ax.legend(loc='lower right', edgecolor='black', fancybox=False, shadow=False)
    ax.text(1.2, 0.75, 'Estimated\nvalue', fontsize=20, horizontalalignment='center')

def plot_empirical_rms_error(ax, data):
    for α, d in data.items():
        if α >= 0.05:
            i = int(α*20)-1
            x = [(70, d[70]-0.01), (35, d[35]-0.01), (0, d[14])]
            ax.plot(d, '-', color='gray', lw=1)
            ax.text(*x[i], f'α={α:.2f}', fontsize=10)
        else:
            i = int(α*100)-1
            style = ['-', '--', '-.', '-']
            x = [(20, d[19]), (35, d[32]), (89, d[85]), (60, d[61]+0.002)]
            ax.plot(d, style[i], color='black', lw=0.8)
            ax.text(*x[i], f'α={α}', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_bounds(-1, 100)
    ax.set_xlim((-1, 100.1))
    ax.set_xlabel('Walks / Episodes', fontsize=15)
    ax.text(60, 0.225, 'Empirical RMS error,\naveraged over states', fontsize=20, horizontalalignment='center', verticalalignment='center')
    ax.text(10, 0.04, 'TD', color='gray', fontsize=15, horizontalalignment='center')
    ax.text(55, 0.15, 'MC', fontsize=15, horizontalalignment='center')

def plot_batch_rms_error(ax, data):
    ax.plot(range(1,101), data['TD'], '-', color='#007ba9', lw=1)
    ax.plot(range(1,101), data['MC'], '-', color='black', lw=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_bounds(0, 100)
    ax.spines['left'].set_bounds(0, 0.25)
    ax.set_xlim((0, 100.1))
    ax.set_ylim((0, 0.25))
    ax.set_xlabel('Walks / Episodes', fontsize=15)
    ax.set_ylabel('RMS error\naveraged\nover states', fontsize=15, rotation=0, labelpad=50, verticalalignment='center')
    ax.text(22, 0.04, 'TD', color='#007ba9', fontsize=15)
    ax.text(50, 0.1, 'MC', fontsize=15)
    ax.text(45, 0.22, 'BATCH TRAINING', fontsize=18)

def exp_6_2():
    np.random.seed(2017)
    episodes = runs = 100

    for r in range(runs):
        TD_agents = [TD(α=0.05*i) for i in range(1,4)]
        MC_agents = [MC(α=0.01*i) for i in range(1,5)]

        if r == 0:
            es_data = [{'e':0,'v':TD_agents[1].V[1:]}]
            rms_data = {}
            for agent in TD_agents+MC_agents:
                rms_data[agent.α] = np.zeros(episodes+1)
                rms_data[agent.α][0] = agent.rms() * runs

        for e in range(1, episodes+1):
            for S, R, S_ in RandomWalk():
                for agent in TD_agents+MC_agents: agent.update(S, R, S_)

            for agent in TD_agents+MC_agents:
                rms_data[agent.α][e] += agent.rms()
            
            if r == 0 and e in [1, 10, 100]:
                es_data.append({'e':e,'v':TD_agents[1].V[1:]})
    else:
        for agent in TD_agents+MC_agents:
            rms_data[agent.α] /= runs

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_estimated_value(axes[0], es_data)
    plot_empirical_rms_error(axes[1], rms_data)
    plt.savefig('exp_6.2_random_walk.png', dpi=300, bbox_inches='tight')

def fig_6_2():
    episodes = runs = 100

    for r in tqdm(range(runs)):
        TD_agent = TD(α=0.001)
        MC_agent = MC(α=0.001)

        if r == 0:
            rms_data = {}
            rms_data['TD'] = np.zeros(episodes)
            rms_data['MC'] = np.zeros(episodes)

        for e in range(episodes):
            for S, R, S_ in RandomWalk():
                TD_agent.update(S, R, S_)
                MC_agent.update(S, R, S_)
            TD_agent.update_batch()
            MC_agent.update_batch()
            rms_data['TD'][e] += TD_agent.rms()
            rms_data['MC'][e] += MC_agent.rms()
    else:
        rms_data['TD'] /= runs
        rms_data['MC'] /= runs

    plt.figure(figsize=(6, 5))
    ax = plt.subplot()
    plot_batch_rms_error(ax, rms_data)
    plt.savefig('fig_6.2_random_walk_batch_training.png', dpi=300, bbox_inches='tight')

# exp_6_2()
# fig_6_2()
