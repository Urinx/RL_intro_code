import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

def fig_8_7():
    braching_factors = [2, 100, 10, 1000, 10000]
    
    def formula(b):
        return [np.sqrt((b-1)/(b*t)) for t in range(1, 2*b+1)]

    plt.figure(figsize=(6, 3))
    ax = plt.subplot()
    for b in braching_factors:
        ax.plot(np.arange(1,2*b+1)/b, formula(b), lw=.5)
    ax.text(0.34, 0.8, 'sample\nupdates', horizontalalignment='center')
    arrows = [(.25, .75, -.18, -.61), (.3, .75, -.13, -.48), (.35, .75, -.01, -.2), (.43, .75, .05, -.04)]
    for x, y, dx, dy in arrows: ax.arrow(x, y, dx, dy, lw=.5, head_length=.02, head_width=.01, length_includes_head=True, fc='k')
    labels = [(.66, .65, '$b=2$ (branching factor)'), (.45, .46, '$b=10$'), (.23, .23, '$b=100$'), (.11, .11, '$b=1000$'), (.07, .04, '$b=10000$')]
    for x, y, label in labels: ax.text(x, y, label, fontsize=8 if x>.1 else 5)
    ax.plot([0, 1, 1, 2], [1, 1, 0, 0], '-', color='gray', lw=.5)
    ax.text(1.01, 0.8, 'expected\nupdates')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_bounds(-0.02, 1)
    plt.xlim((-0.02, 2))
    plt.ylim((-0.02, 1.01))
    plt.xticks([0, 1, 2], ['0', '1b', '2b'])
    plt.yticks([0, 1])
    plt.xlabel('Number of       $Q(s\', a\')$ computations')
    plt.text(0.768, -0.206, r"$\max_{a'}$", fontsize=8)
    plt.ylabel('RMS error\nin value\nestimate', labelpad=20, rotation=0, verticalalignment='center')
    plt.savefig('fig_8.7_expected_vs_sample_update.png', dpi=300, bbox_inches='tight')

ACT_N = 2

class Env:
    def __init__(self, states, b):
        self.b = b
        self.S = 0
        self.T = states
        self.P = .1
        self.M = np.random.randint(states, size=(states, ACT_N, b))
        self.R = np.random.randn(states, ACT_N, b)

    def step(self, S, A):
        if np.random.rand() < self.P:
            S_, R = self.T, 0
        else:
            i = np.random.randint(self.b)
            S_ = self.M[S, A, i]
            R = self.R[S, A, i]
        return S_, R

class GreedyPolicy:
    def __init__(self, env, eval_step, eval_runs):
        self.ε = .1
        self.env = env
        self.eval_step = eval_step
        self.eval_runs = eval_runs
        self.reset()

    def reset(self):
        self.Q = np.zeros((self.env.T, ACT_N))

    def act(self, S):
        if np.random.rand() < self.ε:
            return np.random.randint(ACT_N)
        else:
            return np.random.choice([i for i in range(ACT_N) if self.Q[S, i] == self.Q[S].max()])

    def eval_Vπ(self, start_state):
        env = self.env
        rewards = 0
        for r in range(self.eval_runs):
            S = start_state
            while S != env.T:
                A = self.act(S)
                S_, R = env.step(S, A)
                S = S_
                rewards += R
        return rewards / self.eval_runs

class Uniform(GreedyPolicy):
    def run_updates(self, updates):
        Q, env, eval_step = self.Q, self.env, self.eval_step
        V = [0]
        for t in range(updates):
            S = t // ACT_N % env.T
            A = t % ACT_N
            S_ = env.M[S, A]
            Q[S, A] = (1 - env.P) * (env.R[S, A] + np.max(Q[S_, :], axis=1)).mean()
            if (t+1) % eval_step == 0: V.append(self.eval_Vπ(env.S))
            # if (t+1) % eval_step == 0: V.append(Q[env.S].mean())
        return V

class OnPolicy(GreedyPolicy):
    def run_updates(self, updates):
        Q, env, eval_step = self.Q, self.env, self.eval_step
        V = [0]
        S = env.S
        for t in range(updates):
            A = self.act(S)
            S_ = env.M[S, A]
            Q[S, A] = (1 - env.P) * (env.R[S, A] + np.max(Q[S_, :], axis=1)).mean()
            S_, _ = env.step(S, A)
            S = env.S if S_ == env.T else S_
            if (t+1) % eval_step == 0: V.append(self.eval_Vπ(env.S))
            # if (t+1) % eval_step == 0: V.append(Q[env.S].mean())
        return V

def one_run_proc(params):
    agent, env, eval_step, eval_runs, updates = params
    return agent(env, eval_step, eval_runs).run_updates(updates)

def set_ax(ax, states, updates):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim((-updates//50, updates))
    ax.legend(['uniform', 'on-policy'], fontsize=6, loc='lower right')
    ax.set_xlabel('Computation time, in expected updates', fontsize=8)
    ax.set_ylabel('Value of\nstart state\nunder\ngreedy\npolicy', fontsize=8, labelpad=25, rotation=0, verticalalignment='center')

    if states == 1000:
        ax.text(3000, 7, 'b=1')
        ax.text(5000, 3.2, 'b=3')
        ax.text(8000, 1.8, 'b=10')
        y = 5.8
    else:
        ax.text(50000, 4.5, 'b=1')
        y = 4
    ax.text(updates//4*3, y, f'{states} STATES')

def fig_8_8():
    runs = 400
    envs = [(1000, 20000, 100, [1, 3, 10]), (10000, 200000, 1000, [1])]
    agents = [Uniform, OnPolicy]
    workers = 100
    eval_runs = 1000

    Vπs = []
    for states, updates, eval_step, branching_factors in envs:
        Vπ = np.zeros((len(branching_factors), len(agents), updates // eval_step +1))
        for i, b in enumerate(branching_factors):
            env = Env(states, b)
            for j, agent in enumerate(agents):
                print(states, b, agent)
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    for v in executor.map(one_run_proc, [(agent, env, eval_step, eval_runs, updates)]*runs):
                        Vπ[i, j] += v
        Vπ /= runs
        Vπs.append(Vπ)

    plt.figure(figsize=(3, 7))
    for i in range(len(Vπs)):
        Vπ, (states, updates, eval_step, branching_factors) = Vπs[i], envs[i]
        ax = plt.subplot(2, 1, i+1)
        for j, b in enumerate(branching_factors):
            ax.plot(np.arange(updates//eval_step+1)*eval_step, Vπ[j, 0], 'k-', lw=.5)
            ax.plot(np.arange(updates//eval_step+1)*eval_step, Vπ[j, 1], 'k-', lw=.9)
        set_ax(ax, states, updates)
    plt.subplots_adjust(wspace=0, hspace=.3)
    plt.savefig('fig_8.8_trajectory_sampling.png', dpi=300, bbox_inches='tight')

# fig_8_7()
# fig_8_8()