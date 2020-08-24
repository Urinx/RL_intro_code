'''
Dyna Maze

Author: Urinx
'''
import heapq
import random
import collections
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.table import Table
from concurrent.futures import ProcessPoolExecutor

class Maze:
    H = 6
    W = 9
    S = (2, 0)
    G = (0, 8)
    OBSTACLE = set([(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)])
    ACTS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ACT_NUM = 4

    @classmethod
    def step(self, S, A):
        A = self.ACTS[A]
        S_ = (
            min(max(S[0]+A[0], 0), self.H-1),
            min(max(S[1]+A[1], 0), self.W-1)
        )
        if S_ in self.OBSTACLE: S_ = S
        R = 1 if self.end(S_) else 0
        return S_, R

    @classmethod
    def end(self, S):
        return S == self.G

class BlockingMaze(Maze):
    S = (5, 3)
    G = (0, 8)
    OBSTACLE = set([(3,0), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7)])
    OBSTACLE2 = set([(3,8), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7)])

    @classmethod
    def change(self):
        self.OBSTACLE, self.OBSTACLE2 = self.OBSTACLE2, self.OBSTACLE

class ShortcutMaze(BlockingMaze):
    OBSTACLE = set([(3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7), (3,8)])
    OBSTACLE2 = set([(3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7)])

class CoarseMaze(Maze):
    @classmethod
    def set_resolution(self, f=0):
        fh, fw = 2**(f//2), 2**(f//2+f%2)
        self.F = (fh, fw)
        self.H = Maze.H * fh
        self.W = Maze.W * fw
        self.S = (Maze.S[0]*fh, Maze.S[1]*fw)

        def extend(states):
            a = set()
            for h, w in states:
                for i in range(fh):
                    for j in range(fw):
                        a.add((h*fh+i, w*fw+j))
            return a

        self.G = extend([Maze.G])
        self.OBSTACLE = extend(Maze.OBSTACLE)
        self.SIZE = self.H*self.W-len(self.OBSTACLE)

    @classmethod
    def end(self, S):
        return S in self.G

    @classmethod
    def check_path(self, agent):
        fh, fw = self.F
        max_steps = (5*fh + 8*fw + 1) * 1.2 # relaxed optifmal path
        steps = 0
        S = self.S
        while not self.end(S):
            A = agent.π_(S)
            S, _ = self.step(S, A)
            steps += 1
            if steps > max_steps: return False
        return True

class Dyna_Q:
    def __init__(self, α=.1, ε=.1, γ=.95, n=0, κ=0):
        self.α = α
        self.ε = ε
        self.γ = γ
        self.n = n
        self.κ = κ
        self.t = 0
        self.method = 'Dyna-Q' if κ==0 else 'Dyna-Q+'
        self.reset()

    def reset(self):
        self.Q = collections.defaultdict(lambda:[0]*Maze.ACT_NUM)
        self.M = {}
        self.t = 0
        self.backups = 0

    def act(self, S):
        if random.random() >= self.ε:
            A = random.choice([a for a, v in enumerate(self.Q[S]) if v==max(self.Q[S])])
        else:
            A = random.randint(0, Maze.ACT_NUM-1)
        return A

    def update(self, S, A, R, S_):
        α, γ, κ, Q, M = self.α, self.γ, self.κ, self.Q, self.M

        # direct reinforcement learning
        Q[S][A] += α * (R + γ * max(Q[S_]) - Q[S][A])

        # model learning
        self.t += 1
        # Actions that had never been tried were allowed to be considered in the planning step
        if κ != 0 and (S, A) not in M:
            for a in range(Maze.ACT_NUM):
                M[S, a] = (S, 0, 1)
        M[S, A] = (S_, R, self.t)

        # planning
        for _ in range(self.n):
            S, A = random.choice(list(M.keys()))
            S_, R, t = M[S, A]
            if κ:
                τ = self.t - t
                R += κ * np.sqrt(τ)
            Q[S][A] += α * (R + γ * max(Q[S_]) - Q[S][A])

        self.backups += self.n + 1

    @property
    def π(self):
        H, W = map(max, zip(*self.Q.keys()))
        π = -np.ones((H+1, W+1), dtype=int)
        for S in self.Q:
            if max(self.Q[S]) != 0:
                π[S] = self.π_(S)
        return π

    def π_(self, S):
        return np.argmax(self.Q[S])

    def __str__(self):
        s = f'[{self.method}] '
        s += ', '.join(f'{k}: {v}' for k, v in self.__dict__.items() if type(v) in [int, float])
        s += f', π:\n{self.π}'
        return s

class QueueDyna(Dyna_Q):
    '''
    aka Prioritized Sweeping
    '''
    def __init__(self, α=.1, ε=.1, γ=.95, n=5, θ=.0001):
        super().__init__(α, ε, γ, n)
        self.θ = θ
        self.method = 'queue-Dyna'
        self.reset()

    def reset(self):
        super().reset()
        self.queue = []
        self.lead_to = collections.defaultdict(set)

    def update(self, S, A, R, S_):
        α, γ, θ, Q, M, L2 = self.α, self.γ, self.θ, self.Q, self.M, self.lead_to

        # insert to queue
        P = abs(R + γ * max(Q[S_]) - Q[S][A])
        if P > θ: heapq.heappush(self.queue, (-P, (S, A)))

        # model learning
        M[S, A] = (S_, R, -P)
        L2[S_].add((S, A, R))

        # planning
        for _ in range(self.n):
            while 1: # skip duplicate items
                if not self.queue: return
                p1, (S, A) = heapq.heappop(self.queue)
                S_, R, p2 = M[S, A]
                if p1 == p2:
                    M[S, A] = (S_, R, None)
                    break
            Q[S][A] += α * (R + γ * max(Q[S_]) - Q[S][A])

            for Ś, Á, R in L2[S]:
                P = abs(R + γ * max(Q[S]) - Q[Ś][Á])
                if P > θ:
                    heapq.heappush(self.queue, (-P, (Ś, Á)))
                    M[Ś, Á] = (S, R, -P)

            self.backups += 1

def plot_maze(ax):
    center = (60, 750)
    ax.arrow(*center, 0, 60, head_width=.5, head_length=10, length_includes_head=True, fc='k')
    ax.arrow(*center, 0, -60, head_width=.5, head_length=10, length_includes_head=True, fc='k')
    ax.arrow(*center, 3.5, 0, head_width=8, head_length=.6, length_includes_head=True, fc='k')
    ax.arrow(*center, -3.5, 0, head_width=8, head_length=.6, length_includes_head=True, fc='k')
    ax.text(center[0], center[1]-100, 'actions', fontsize=10, horizontalalignment='center')

    H = 0.6
    Δh, Δw = 1/Maze.H, 1/Maze.W
    tb = Table(ax, bbox=[20/65, H, 30/65, 1-H])
    for i in range(Maze.H):
        for j in range(Maze.W):
            t = ''
            if (i, j) == Maze.S: t = 'S'
            elif (i, j) == Maze.G: t = 'G'
            c = tb.add_cell(i, j, Δw, Δh, text=t, loc='center')
            c.set_lw(.5)
            c.set_fontsize(14)
            if (i, j) in Maze.OBSTACLE:
                c.set_fc('#aaaaaa')
    ax.add_table(tb)

def plot_result(ax, data):
    colors = ['#666666', '#aaaaaa', 'k']
    for i, c in enumerate(colors):
        ax.plot(range(2, 51), data[i, 1:], '-', lw=1, color=c)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_bounds(1, 50)
    ax.spines['left'].set_bounds(-10, 820)
    ax.set_xlim((1, 65))
    ax.set_ylim((-10, 860))
    ax.set_xticks([2, 10, 20, 30, 40, 50])
    ax.set_yticks([14, 200, 400, 600, 800])
    ax.set_xlabel('Episodes', fontsize=15)
    ax.set_ylabel('Steps\nper\nepisode', fontsize=15, rotation=0, labelpad=40)

    c = '#232323'
    ax.annotate('0 planning steps\n(direct RL only)',
        xy=(5, data[0, 4]), xytext=(12, 400), horizontalalignment='center',
        arrowprops=dict(arrowstyle='-|>', color=c))
    ax.annotate('5 planning steps',
        xy=(3, data[1, 2]), xytext=(20, 300), horizontalalignment='center',
        arrowprops=dict(arrowstyle='-|>', color=c))
    ax.annotate('50 planning steps',
        xy=(3, data[2, 2]), xytext=(30, 200), horizontalalignment='center',
        arrowprops=dict(arrowstyle='-|>', color=c))

def fig_8_2():
    random.seed(3)
    runs = 30
    episodes = 50
    agents = [Dyna_Q(n=n) for n in [0, 5, 50]]
    data = np.zeros((len(agents), episodes))

    for i, agent in enumerate(agents):
        for r in tqdm(range(runs)):
            agent.reset()
            for e in range(episodes):
                steps = 0
                S = Maze.S
                while S != Maze.G:
                    A = agent.act(S)
                    S_, R = Maze.step(S, A)
                    agent.update(S, A, R, S_)
                    S = S_
                    steps += 1
                data[i, e] += steps
    else:
        data /= runs

    plt.figure(figsize=(8, 6))
    ax = plt.subplot()
    plot_result(ax, data)
    plot_maze(ax)
    plt.savefig('fig_8.2_dyna_maze.png', dpi=300, bbox_inches='tight')

def plot_policy(ax, title, data):
    _, S, policy = data
    ax.set_axis_off()
    ax.set_title(title)

    Δh, Δw = 1/Maze.H, 1/Maze.W
    tb = Table(ax, bbox=[0, 0, 1, 1])
    direction = '↑↓←→ '
    for i in range(Maze.H):
        for j in range(Maze.W):
            t = ''
            if (i, j) == Maze.S: t = 'S'
            elif (i, j) == Maze.G: t = 'G'
            elif (i, j) == S: t = '◼'
            else:
                t = direction[policy[i,j]]
            c = tb.add_cell(i, j, Δw, Δh, text=t, loc='center')
            c.set_lw(.5)
            c.set_fontsize(14)
            if (i, j) in Maze.OBSTACLE:
                c.set_fc('#aaaaaa')
    ax.add_table(tb)

def fig_8_3():
    random.seed(2840) # I have trid all seeds less than 10000 and find this one
    agents = [Dyna_Q(n=n) for n in [0, 50]]

    for e in range(2):
        policy = []
        for agent in agents:
            p = []
            steps = 0
            S = Maze.S
            while S != Maze.G:
                A = agent.act(S)
                S_, R = Maze.step(S, A)
                agent.update(S, A, R, S_)
                p.append((steps, S, agent.π.copy()))
                S = S_
                steps += 1
            policy.append(p[steps//2])

    plt.figure(figsize=(9, 3))
    plot_policy(plt.subplot(121), 'WITHOUT PLANNING (n=0)', policy[0])
    plot_policy(plt.subplot(122), 'WITH PLANNING (n=50)', policy[1])
    plt.savefig('fig_8.3_dyna_maze_policy.png', dpi=300, bbox_inches='tight', pad_inches=.5)
    plt.close()

def run_maze_test(agents, maze, runs, steps, change_step):
    data = np.zeros((len(agents), steps+1))
    for r in tqdm(range(runs)):
        for i, agent in enumerate(agents):
            agent.reset()
            rewards = 0
            S = maze.S
            for step in range(steps):
                A = agent.act(S)
                S_, R = maze.step(S, A)
                agent.update(S, A, R, S_)
                S = S_ if S_ != maze.G else maze.S
                rewards += R
                data[i, step+1] += rewards
                if step == change_step: maze.change()
            maze.change()
    else:
        data /= runs
    return data

def plot_change_maze(out, data, agents, maze, change_step, xlim, ylim, xticks, yticks):
    plt.figure(figsize=(6, 6))
    ax = plt.subplot()

    for i in range(data.shape[0]):
        ax.plot(range(data.shape[1]), data[i], 'k-', lw=1)
        x = data.shape[1]//5*4
        ha, va = [('right', 'bottom'), ('left', 'top')][i]
        ax.text(x, data[i][x], agents[i].method, horizontalalignment=ha, verticalalignment=va)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_bounds(xlim[0], xlim[1])
    ax.spines['left'].set_bounds(ylim[0], ylim[1])
    ax.set_xlim((xlim[0], xlim[2]))
    ax.set_ylim((ylim[0], ylim[2]))
    ax.set_xticks(xticks)
    ax.set_yticks(yticks[0])
    ax.set_yticklabels(yticks[1])
    ax.set_xlabel('Time steps', fontsize=15, labelpad=15)
    ax.text(-xlim[1]//5, ylim[1]//2, 'Cumulative\nreward', fontsize=15, verticalalignment='center', horizontalalignment='center')

    ax.plot([change_step, change_step, xlim[1]//2, xlim[1]//2], [ylim[0], ylim[1], ylim[2]*.75, ylim[2]], 'k--', lw=.6)

    Δh, Δw = 1/maze.H, 1/maze.W
    tb = Table(ax, bbox=[abs(xlim[0])/(abs(xlim[0])+xlim[2]), .75, .4, .25])
    tb2 = Table(ax, bbox=[.6, .75, .4, .25])
    for i in range(maze.H):
        for j in range(maze.W):
            t = ''
            if (i, j) == maze.S: t = 'S'
            elif (i, j) == maze.G: t = 'G'
            else: t = ''
            c = tb.add_cell(i, j, Δw, Δh, text=t, loc='center')
            c.set_lw(.5)
            c.set_fontsize(10)
            c2 = tb2.add_cell(i, j, Δw, Δh, text=t, loc='center')
            c2.set_lw(.5)
            c2.set_fontsize(10)
            if (i, j) in maze.OBSTACLE: c.set_fc('#aaaaaa')
            if (i, j) in maze.OBSTACLE2: c2.set_fc('#aaaaaa')
    ax.add_table(tb)
    ax.add_table(tb2)

    plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=.3)
    plt.close()

def fig_8_4():
    runs = 100
    steps = 3000
    change_step = 1000
    agents = [Dyna_Q(α=1, n=20, κ=1e-4), Dyna_Q(α=1, n=20)]
    data = run_maze_test(agents, BlockingMaze, runs, steps, change_step)

    xlim, ylim = (-40, 3000, 3000), (-3, 150, 230)
    xticks, yticks = [0, 1000, 2000, 3000], [[0, 50, 100, 150], [0, '', '', 150]]
    plot_change_maze('fig_8.4_dyna_on_blocking_maze.png', data, agents, BlockingMaze, change_step, xlim, ylim, xticks, yticks)

def fig_8_5():
    runs = 100
    steps = 6000
    change_step = 3000
    agents = [Dyna_Q(α=1, n=50, κ=1e-3), Dyna_Q(α=1, n=50)]
    data = run_maze_test(agents, ShortcutMaze, runs, steps, change_step)

    xlim, ylim = (-140, 6000, 6000), (-10, 400, 600)
    xticks, yticks = [0, 3000, 6000], [[0, 100, 200, 300, 400], [0, '', '', '', 400]]
    plot_change_maze('fig_8.5_dyna_on_shortcut_maze.png', data, agents, ShortcutMaze, change_step, xlim, ylim, xticks, yticks)

def one_run_proc(i):
    agent = QueueDyna(α=.5, γ=.95, n=5, θ=.0001) if i else Dyna_Q(α=.5, γ=.95, n=5)
    while not CoarseMaze.check_path(agent):
        S = CoarseMaze.S
        while not CoarseMaze.end(S):
            A = agent.act(S)
            S_, R = CoarseMaze.step(S, A)
            agent.update(S, A, R, S_)
            S = S_
    return agent.backups

def exp_8_4():
    '''
    Peng, J., Williams, R. J. (1993). Efficient learning and planning within the Dyna framework.
    Adaptive Behavior, 1(4):437–454
    '''
    runs = 100
    max_resolution = 7
    workers = 100
    
    data = np.zeros((2, max_resolution+1))
    data[:, 0] = 1e3
    gridsize = []

    for f in tqdm(range(max_resolution)):
        CoarseMaze.set_resolution(f)
        gridsize.append(CoarseMaze.SIZE)

        for i in range(2):
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for backups in executor.map(one_run_proc, [i]*runs):
                    data[i, f+1] += backups
            # print(f'f: {f}, agent: {i}, backups: {data[i, f+1]/runs}')
    else:
        data /= runs

    plt.figure(figsize=(6, 6))
    ax = plt.subplot()
    ax.plot(data[0], 'k-', lw=.6, label='Dyna-Q')
    ax.plot(data[1], 'k-', lw=1, label='Queue Dyna')
    ax.text(4.8, data[0,5], 'Dyna-Q', fontsize=15, verticalalignment='bottom', horizontalalignment='right')
    ax.text(4, data[1,3], 'prioritized\nsweeping', fontsize=15, verticalalignment='top', horizontalalignment='center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim((0, len(gridsize)))
    ax.set_xticklabels([0]+gridsize)
    ax.set_yscale('log')
    ax.set_ylim((10, 1.5e6))
    ax.set_xlabel('Gridworld size (#states)', fontsize=15, labelpad=15)
    ax.set_ylabel('Updates\nuntil\noptimal\nsolution', fontsize=15, labelpad=50, rotation=0, verticalalignment='center')
    plt.savefig('exp_8.4_prioritized_sweeping.png', dpi=300, bbox_inches='tight', pad_inches=.5)
    plt.close()

# fig_8_2()
# fig_8_3()
# fig_8_4()
# fig_8_5()
# exp_8_4()
