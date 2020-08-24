"""
The 10-armed Testbed

To roughly assess the relative effectiveness of the
greedy and epsilon-greedy action-value methods

Author: Urinx
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm import tqdm, trange
from datetime import datetime

n_run, n_step = 2000, 1000
k, μ, σ, ν = 10, 0, 1, 1

class BanditAgent:
    def __init__(self, k, /, method='Egreedy', *, ε=0.1, q=0, α=0.1, c=1, with_baseline=True):
        self.α = α # constant step-size, if it's 0 then apply sample-average, eg. 1/n
        self.init_q = q # initial action-value
        self.ε = ε # exploration rate
        self.k = k # action sapce
        self.c = c # confidence
        self.method = method # Egreedy, UCB, Random, Gradient
        self.with_baseline = with_baseline
        self.reset()

    def __str__(self):
        if self.method == 'Egreedy':
            return f'[Egreedy Agent] ε: {self.ε}, init Q: {self.init_q}, α: {self.α}'
        elif self.method == 'UCB':
            return f'[UCB Agent] c: {self.c}, init Q: {self.init_q}, α: {self.α}'
        elif self.method == 'Random':
            return '[Random Agent]'
        elif self.method == 'Gradient':
            return f'[GradAscent Agent] with baseline: {self.with_baseline}, α: {self.α}'
        else:
            return '[Unknown Agent]'

    def reset(self):
        self.t = 0
        self.Q = [self.init_q] * self.k
        self.N = [0] * self.k
        self.H = [0] * self.k

    def act(self):
        if self.method == 'Egreedy':
            if np.random.rand() < self.ε:
                return np.random.randint(self.k)
            else:
                return np.random.choice(np.where(self.Q == np.max(self.Q))[0])
        elif self.method == 'Random':
            return np.random.randint(self.k)
        elif self.method == 'UCB':
            not_ever_selected = [i for i in range(self.k) if self.N[i] == 0]
            if len(not_ever_selected):
                return not_ever_selected[0]
            else:
                A = [self.Q[a] + self.c * np.sqrt(np.log(self.t) / self.N[a]) for a in range(self.k)]
                return np.random.choice(np.where(A == np.max(A))[0])
        elif self.method == 'Gradient':
            p = (eh := np.exp(self.H)) / eh.sum()
            return np.random.choice(self.k, p=p)

    def update(self, R, A):
        α, k, Q, N, H = self.α, self.k, self.Q, self.N, self.H

        N[A] += 1
        self.t += 1
        if self.method == 'Gradient':
            Q[A] += (R - Q[A]) / N[A] # -> $\bar R_t$
            p = (eh := np.exp(H)) / eh.sum()

            if self.with_baseline:
                baseline = self.Q
            else:
                baseline = [0] * k

            for x in range(k):
                H[x] += α * (R - baseline[x]) * ((1 if A==x else 0) - p[x])
        else:
            if α == 0:
                Q[A] += (R - Q[A]) / N[A]
            else:
                Q[A] += α * (R - Q[A])

def run_k_bandit_test(agents, μ=μ):
    n_agent = len(agents)
    avg_reward_per_step = np.zeros((n_agent, n_step))
    optimal_action_ratio = np.zeros((n_agent, n_step))

    start_t = datetime.now()
    for _ in trange(n_run, leave=False):
        q_true = np.random.normal(μ, σ, size=k)
        optimal_action = np.argmax(q_true)

        for agent in agents: agent.reset()

        for n in range(n_step):
            for i, agent in enumerate(agents):
                A = agent.act()
                R = np.random.normal(q_true[A], ν)
                avg_reward_per_step[i, n] += R
                optimal_action_ratio[i, n] += int(A == optimal_action)
                agent.update(R, A)

    avg_reward_per_step /= n_run
    optimal_action_ratio /= n_run
    end_t = datetime.now()

    return avg_reward_per_step, optimal_action_ratio, end_t - start_t

def set_fig(ax, ylabel, /, ymax, xticks, yticks, *, labelpad=30, percent=False):
    plt.ylabel(ylabel, rotation=0, labelpad=labelpad, fontsize=15, verticalalignment='center')
    plt.xlabel('Steps', fontsize=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim((-20, 1000))
    ax.spines['bottom'].set_bounds(-20, 1000)
    plt.xticks(xticks)

    ax.set_ylim((0, ymax))
    ax.spines['left'].set_bounds(0, ymax)
    plt.yticks(yticks)
    if percent:
        yticks = mtick.FuncFormatter(lambda x,_: f'{x*100:.0f}%')
        plt.gca().yaxis.set_major_formatter(yticks)

def fig_2_1():
    np.random.seed(0)
    q_true = np.random.normal(μ, σ, size=k)
    print(f'[*] One set of randomly generated k-armed bandit with Q value:\n{q_true}')
    data = [np.random.normal(q, σ, size=100) for q in q_true]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca()
    ax.plot([0.5, 10.5], [0, 0], '--', linewidth=1, color='k')
    violin_parts = ax.violinplot(data, showextrema=False)
    
    for vp in violin_parts['bodies']:
        vp.set_facecolor('#BBBBBB')
        vp.set_alpha(1)
    for i in range(k):
        ax.axvline(i+1, ymin=.03, ymax=0.97, color='#BBBBBB', linewidth=1, linestyle='-')
        ax.plot([i+.7, i+1.3], [q_true[i], q_true[i]], color='k', linewidth=1)
        ax.text(i+1.3, q_true[i], '$q_{*}('+str(i+1)+')$', verticalalignment='center')

    plt.xlabel('Action', fontsize=15, labelpad=10)
    plt.ylabel('Reward\ndsitribution', rotation=0, labelpad=40, fontsize=15)
    plt.xticks(np.arange(1, k+1, 1), fontsize=12)
    plt.yticks(np.arange(-3, 4, 1), fontsize=12)
    
    ax.tick_params(left=False)
    ax.tick_params(axis='x', length=6, width=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlim((0, 11))
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_bounds(1, 10)

    plt.savefig('fig2.1-k_armed_bandit_Q.png', dpi=300, bbox_inches='tight')
    plt.close()

def fig_2_2():
    print(f'[*] Run {k}-armed testbed')
    print(f'[*] Repeating {n_run} independent runs')
    print(f'[*] Action value selected from normal distribution with mean {μ} and variance {σ}')
    print(f'[*] Reward selected from normal distribution with mean q* and variance {ν}')
    print(f'[*] Measure performance over {n_step} time steps')

    ε_arr = [0, 0.01, 0.1]
    agents = [BanditAgent(k, method='Egreedy', ε=ε, α=0) for ε in ε_arr]
    avg_reward_per_step, optimal_action_ratio, cost_time = run_k_bandit_test(agents)

    print(f'[*] Cost time: {cost_time}')
    print(f'[*] Agent ε-greedy: {ε_arr}')
    print(f'[*] Average reward: {avg_reward_per_step[:, -1]}')
    print(f'[*] Optimal action ratio: {optimal_action_ratio[:, -1]}')

    plt.figure(figsize=(8, 8))
    color = ['#096347','#BE3033','#0B1015']
    config = [(avg_reward_per_step, 'Average\nreward', 1.5, np.arange(0, 1.6, 0.5), False), (optimal_action_ratio, '%\nOptimal\naction', 1, np.arange(0, 1.1, 0.2), True)]
    for i in range(2):
        data, ylabel, ymax, yticks, percent = config[i]
        ax = plt.subplot(2, 1, i+1)
        for j, ε in enumerate(ε_arr): plt.plot(data[j], label=f'$\epsilon={ε}$', color=color[j], lw=1)
        plt.legend(loc='lower right')
        set_fig(ax, ylabel, ymax, [1, 250, 500, 750, 1000], yticks, labelpad=40, percent=percent)

    plt.savefig('fig2.2-k_armed_bandit_result.png', dpi=300, bbox_inches='tight')
    plt.close()

def fig_2_3():
    conf = [{'q': 5, 'ε': 0, 'α': 0.1}, {'q': 0, 'ε': 0.1, 'α': 0.1}]
    agents = [BanditAgent(k, method='Egreedy', **c) for c in conf]
    print(f'[*] Agent:')
    for agent in agents: print(f' └──{agent}')

    _, optimal_action_ratio, cost_time = run_k_bandit_test(agents)
    print(f'[*] Cost time: {cost_time}')
    print(f'[*] Optimal action ratio: {optimal_action_ratio[:, -1]}')

    plt.figure(figsize=(8, 4))
    ax = plt.subplot()
    color = ['black', 'gray']
    for i, c in enumerate(conf): plt.plot(optimal_action_ratio[i], color=color[i], lw=1)
    plt.text(250, .8, 'optimistic, greedy\n$Q_1=5, \epsilon=0$', fontsize=12, horizontalalignment='center')
    plt.text(760, .55, 'realistic, ε-greedy\n$Q_1=5, \epsilon=0$', fontsize=12, horizontalalignment='center')
    set_fig(ax, '%\nOptimal\naction', 1, [1] + [200*(i+1) for i in range(5)], [.2*i for i in range(6)], percent=True)
    plt.savefig('fig_2.3_optimistic_initial_values.png', dpi=300, bbox_inches='tight')
    plt.close()

def fig_2_4():
    agents = [
        BanditAgent(k, method='UCB', c=2, α=0),
        BanditAgent(k, method='Egreedy', ε=0.1, α=0)
    ]
    print(f'[*] Agent:')
    for agent in agents: print(f' └──{agent}')

    avg_reward_per_step, _, cost_time = run_k_bandit_test(agents)
    print(f'[*] Cost time: {cost_time}')
    print(f'[*] Average reward: {avg_reward_per_step[:, -1]}')

    plt.figure(figsize=(8, 4))
    ax = plt.subplot()
    color = ['#BCBCBC', '#385CFF']
    for i, (x, y, label) in enumerate([(600, 1.2, 'ε-greedy $\epsilon=0.1$'), (200, 1.5, 'UCB $c=2$')]):
        plt.plot(avg_reward_per_step[1-i], label=label, color=color[i], lw=1)
        plt.text(x, y, label, color=color[i], fontsize=12)

    set_fig(ax, 'Average\nreward', 1.6, [1, 250, 500, 750, 1000], [0, .5, 1, 1.5])
    ax.set_ylim((-.1, 1.6))
    ax.spines['left'].set_bounds(-.1, 1.6)
    plt.savefig('fig_2.4_average_performance_of_UCB.png', dpi=300, bbox_inches='tight')
    plt.close()

def fig_2_5():
    agents = [
        BanditAgent(k, method='Gradient', α=0.1, with_baseline=True),
        BanditAgent(k, method='Gradient', α=0.4, with_baseline=True),
        BanditAgent(k, method='Gradient', α=0.1, with_baseline=False),
        BanditAgent(k, method='Gradient', α=0.4, with_baseline=False)
    ]
    print(f'[*] Agent:')
    for agent in agents: print(f' └──{agent}')

    _, optimal_action_ratio, cost_time = run_k_bandit_test(agents, μ=4)
    print(f'[*] Cost time: {cost_time}')
    print(f'[*] Optimal action ratio: {optimal_action_ratio[:, -1]}')

    plt.figure(figsize=(8, 4))
    ax = plt.subplot()
    color = ['#0433FF', '#758EFF', '#945200', '#BF9766']
    for i in range(len(agents)):
        plt.plot(optimal_action_ratio[i], color=color[i], lw=1)
        if (α := agents[i].α) == .1: x, y = 250, optimal_action_ratio[i, 250]+.05
        else: x, y = 800, optimal_action_ratio[i, 800]-.05
        plt.text(x, y, f'{α = }', fontsize=13, color=color[i])
    set_fig(ax, '%\nOptimal\naction', 1, [1, 250, 500, 750, 1000], [.2*i for i in range(6)], percent=True)
    plt.text(650, .67, 'with baseline', fontsize=13, horizontalalignment='center', color=color[0])
    plt.text(650, .35, 'without baseline', fontsize=13, horizontalalignment='center', color=color[2])
    plt.savefig('fig_2.5_gradient_with_and_without_baseline.png', dpi=300, bbox_inches='tight')
    plt.close()

def fig_2_6():
    e_greedy_agents = [BanditAgent(k, method='Egreedy', α=0, ε=2**e) for e in range(-7, -1)]
    gradient_agents = [BanditAgent(k, method='Gradient', α=2**a, with_baseline=True) for a in range(-5, 3)]
    ucb_agents = [BanditAgent(k, method='UCB', c=2**c, α=0) for c in range(-4, 3)]
    optim_init_agents = [BanditAgent(k, method='Egreedy', α=0.1, q=2**q, ε=0) for q in range(-2, 3)]
    agents = e_greedy_agents + gradient_agents + ucb_agents + optim_init_agents
    print(f'[*] Agent ({len(agents)}):')
    for agent in agents: print(f' └──{agent}')

    avg_reward_per_step, _, cost_time = run_k_bandit_test(agents)
    avg_reward = avg_reward_per_step.mean(1)
    e_greedy_r = avg_reward[:6]
    gradient_r = avg_reward[6:14]
    ucb_r = avg_reward[14:21]
    optiminit_r = avg_reward[21:]
    print(f'[*] Cost time: {cost_time}')
    print(f'[*] Average reward:')
    print(f'    e-greedy: {e_greedy_r}')
    print(f'    gradient: {gradient_r}')
    print(f'         ucb: {ucb_r}')
    print(f'    opt init: {optiminit_r}')

    plt.figure(figsize=(8, 5))
    ax = plt.subplot()
    data = [
        ('ε-greedy', 'r', range(-7, -1), e_greedy_r, -6, 1.3),
        ('gradient\nbandit', 'g', range(-5, 3), gradient_r, 0, 1.2),
        ('UCB', 'b', range(-4, 3), ucb_r, -2, 1.45),
        ('greedy with\noptimistic\ninitialization\nα=0.1', 'black', range(-2, 3), optiminit_r, 1.8, 1.42)
    ]
    for i, (label, color, x, y, px, py) in enumerate(data):
        plt.plot(x, y, color=color, lw=1)
        plt.text(px, py, label, color=color, fontsize=12, horizontalalignment='center', verticalalignment='center')
    plt.ylabel('Average\nreward\nover first\n1000 steps', rotation=0, labelpad=50, fontsize=15, verticalalignment='center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim((0.99, 1.5))
    ax.spines['left'].set_bounds(0.99, 1.5)
    plt.yticks(np.arange(1, 1.51, 0.1))

    from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox
    # x-axis label
    xbox0 = TextArea(' '*20)
    xbox1 = TextArea(r'$\epsilon$', textprops=dict(color='r', fontsize=15))
    xbox2 = TextArea(r'$\alpha$', textprops=dict(color='g', fontsize=15))
    xbox3 = TextArea(r'$c$', textprops=dict(color='b', fontsize=15))
    xbox4 = TextArea(r'$Q_0$', textprops=dict(color='black', fontsize=15))
    xbox = HPacker(children=[xbox0, xbox1, xbox2, xbox3, xbox4], align='center', pad=-20, sep=10)
    anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0., frameon=False,
                                      bbox_to_anchor=(0.3, -0.07),
                                      bbox_transform=ax.transAxes, borderpad=0.)
    ax.add_artist(anchored_xbox)
    plt.xticks(range(-7, 3))
    ax.set_xticklabels([f'{"1/" if i < 0 else ""}{2**abs(i)}' for i in range(-7, 3)])
    ax.set_xlim((-7.1, 2))
    ax.spines['bottom'].set_bounds(-7.1, 2)

    plt.savefig('fig_2.6_parameter_study_of_various_algorithms.png', dpi=300, bbox_inches='tight')
    plt.close()

# fig_2_1()
# fig_2_2()
# fig_2_3()
# fig_2_4()
# fig_2_5()
# fig_2_6()
