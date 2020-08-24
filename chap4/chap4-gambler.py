'''
Gambler's Problem

Author: Urinx
'''
import numpy as np
import matplotlib.pyplot as plt

class Gambler:
    def __init__(self, goal=100, Ph=0.4, γ=1):
        self.goal = goal
        self.Ph = Ph
        self.γ = γ
        self.V = np.zeros(goal)
        self.π = np.zeros(goal)
        self.k = 0

    def step(self, S, A):
        # coin comes up heads
        S_ = min(S + A, self.goal) % 100
        R = 1 if S_ == 0 else 0
        p = self.Ph
        yield S_, R, p
        # tail
        S_, R, p = S - A, 0, 1 - self.Ph
        yield S_, R, p

    def value_iteration(self, hook=None):
        γ, θ, G, V, π = self.γ, 1e-9, self.goal, self.V, self.π

        while 1:
            Δ = 0
            for S in range(1, G):
                Q = []
                for A in range(min(S, G-S) + 1):
                    q = sum(p * (R + γ * V[S_]) for S_, R, p in self.step(S, A))
                    Q.append((q, A))
                v = max(Q)[0]
                Δ = max(Δ, abs(v - V[S]))
                V[S] = v
            self.k += 1
            if hook is not None: hook(self.k, V)
            if Δ < θ: break

        for S in range(1, G):
            Q = []
            for A in range(min(S, G-S) + 1):
                Q.append(sum(p * (R + γ * V[S_]) for S_, R, p in self.step(S, A)))
            '''
            The [1:] avoid choosing the '0' action which doesn't change state nor exptected returns.
            Since numpy.argmax chooses the first option in case of ties, rounding the near-ties 
            assures the one associated with the smallest action (or bet) is selected. The output 
            of the app now resembles Figure 4.3. in the Sutton/Bartho's book.
            https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/issues/83
            '''
            π[S] = np.argmax(np.round(Q[1:], 5)) + 1

def fig_4_3():
    gb = Gambler()

    arr = []
    def foo(k, V):
        if k in [1, 2, 3, 4]: arr.append(V.copy())

    gb.value_iteration(foo)
    print(f'V(k={gb.k}):\n{gb.V.round(2)}')
    arr.append(gb.V.copy())

    plt.figure(figsize=(8, 12))
    ax = plt.subplot(211)
    lstyle = ['-', '-', '--', '-', ':']
    for i in range(len(arr)):
        plt.plot(range(1,100), arr[i][1:], color='k', linestyle=lstyle[i], linewidth=2/(i+1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim((0, 100))
    ax.set_ylim((-.01, 1))
    ax.set_xlabel('Capital', fontsize=15)
    ax.set_ylabel('Value\nestimates', rotation=0, fontsize=15, labelpad=40)
    ax.annotate('sweep 1', xy=(50, 0.26), xycoords='data', xytext=(60, 0.25), textcoords='data',
                fontsize=15, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    ax.annotate('sweep 2', xy=(25, 0.13), xycoords='data', xytext=(60, 0.12), textcoords='data',
                fontsize=15, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    ax.annotate('sweep 3', xy=(13, 0.04), xycoords='data', xytext=(60, 0.03), textcoords='data',
                fontsize=15, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    ax.annotate('sweep 4', xy=(8, 0.021), xycoords='data', xytext=(15, 0.3), textcoords='data',
                fontsize=15, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    ax.annotate(f'sweep {gb.k}', xy=(5, 0.02), xycoords='data', xytext=(12, 0.5), textcoords='data',
                fontsize=15, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    ax = plt.subplot(212)
    plt.step(range(1,100), gb.π[1:], 'k', where='mid')
    ax.set_xlabel('Capital', fontsize=15)
    ax.set_ylabel('Final\npolicy\n(stake)', rotation=0, fontsize=15, labelpad=40)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig('fig_4.3_gambler_problem.png', dpi=300, bbox_inches='tight')
    plt.close()

fig_4_3()
