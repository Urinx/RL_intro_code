'''
Blackjack

Author: Urinx
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
from tqdm import tqdm

ACTION_HIT = 0
ACTION_STICK = 1
ACTION_BOTH = 2
HUMAN = False
_print = print

def print(*s):
    if HUMAN: _print(*s)

class FixedStickAgent:
    def __init__(self, stick_sum):
        self.stick_sum = stick_sum
        self.V = np.zeros((10, 10))

    def act(self, S, usable_ace):
        if S[1] < self.stick_sum:
            return ACTION_HIT
        else:
            return ACTION_STICK

class RandomAgent:
    def act(self, S, usable_ace):
        return np.random.choice([ACTION_HIT, ACTION_STICK])

class MCESAgent:
    def __init__(self):
        self.π = np.ones((10, 10, 2), dtype=int) * ACTION_BOTH
        # initial policy sticks only on 20 or 21
        self.π[:, -2:] = ACTION_STICK
        self.action_state_value = np.zeros((10, 10, 2, 2))
        self.action_state_count = np.ones((10, 10, 2, 2))

    def act(self, S, usable_ace):
        S = ((S[0]-1)%10, S[1]-12)
        u = int(usable_ace)
        if self.π[S][u] == ACTION_BOTH:
            return np.random.choice([ACTION_HIT, ACTION_STICK])
        return self.π[S][u]

    def update(self, trajectory):
        G = trajectory[-1]
        for i in range(len(trajectory)//4):
            S, usable_ace, A, _ = trajectory[i*4:(i+1)*4]
            S = ((S[0]-1)%10, S[1]-12)
            u = int(usable_ace)

            self.action_state_value[S][u][A] += G
            self.action_state_count[S][u][A] += 1
            Q = self.action_state_value[S][u] / self.action_state_count[S][u]
            if Q[0] == Q[1]:
                self.π[S][u] = ACTION_BOTH
            else:
                self.π[S][u] = Q.argmax()

class BlackJack:
    '''
    A有两种算法，1或者11，算11总点数不超过21时则必须算成11(usable)，否则算作1。

    当玩家点数小于等于11时，当然会毫不犹豫选择要牌，
    所以真正涉及到做选择的状态是12-21点的状态，
    此时庄家亮牌有A-10种情况，再加上是否有11的A(usable ace)，
    所以21点游戏中所有的状态一共只有200个。
    '''
    def __init__(self, human=False):
        a = ['♣','♦','♥','♠']
        b = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
        self.deck = [i+j for i,j in itertools.product(a, b)]

    def reset(self):
        dealer_cards = self.dealt(2)
        player_cards = self.dealt(2)
        s, _ = self.card2state(player_cards)
        while s <= 11:
            player_cards += self.dealt(1)
            s, _ = self.card2state(player_cards)
        return dealer_cards, player_cards

    def dealt(self, n):
        # cards are dealt from an infinite deck (i.e. with replacement)
        idxs = np.random.randint(len(self.deck), size=n)
        cards = [self.deck[i] for i in idxs]
        return cards

    def card2state(self, cards):
        arr = []
        ace_count = 0
        for card in cards:
            if type(card) is not str:
                arr.append(card)
            elif card[1:] in ['J', 'Q', 'K']:
                arr.append(10)
            elif card[1:] == 'A':
                ace_count += 1
                arr.append(11)
            else:
                arr.append(int(card[1:]))
        s = sum(arr)
        while ace_count > 0 and s > 21:
            s -= 10
            ace_count -= 1
        usable_ace = ace_count != 0
        return s, usable_ace

    def check(self, dealer_cards, player_cards):
        s1, _ = self.card2state(dealer_cards)
        s2, _ = self.card2state(player_cards)

        while s1 < 17:
            dealer_cards += self.dealt(1)
            s1, _ = self.card2state(dealer_cards)

        if s1 > 21 or s2 > s1:
            return 1
        elif s2 < s1:
            return -1
        else:
            return 0

    def simulate(self, player, init=None):
        trajectory = []
        is_end = False

        if init is not None:
            s1, s2 = init[0]
            usable_ace = init[1]
            dealer_cards = [s1] + self.dealt(1)
            if usable_ace:
                player_cards = [s2-11, '♥A']
            else:
                player_cards = [s2]
        else:
            dealer_cards, player_cards = self.reset()
            s1, _ = self.card2state(dealer_cards[:1])
            s2, usable_ace = self.card2state(player_cards)
        
        print(f'dealer: {dealer_cards[0]} ({s1})')
        print(f'player: {" ".join(map(str, player_cards))} ({s2}{"↑" if usable_ace else "↓"})')

        while 1:
            s = (s1, s2)
            if trajectory == [] and init is not None and len(init) == 3:
                act = init[2]
            else:
                act = player.act(s, usable_ace)
            r = 0

            trajectory.append(s)
            trajectory.append(usable_ace)
            trajectory.append(act)

            if act == ACTION_HIT:
                player_cards += self.dealt(1)
                s2, usable_ace = self.card2state(player_cards)

                print('act: hit')
                print(f'player: {" ".join(map(str, player_cards))} ({s2}{"↑" if usable_ace else "↓"})')

                # check bust
                if s2 > 21:
                    s1, _ = self.card2state(dealer_cards)
                    print(f'dealer: {" ".join(map(str, dealer_cards))} ({s1})')
                    print('end: go bust')
                    r = -1
                    is_end = True
            elif act == ACTION_STICK:
                print('act: stick')
                result = self.check(dealer_cards, player_cards)
                s1, _ = self.card2state(dealer_cards)
                print(f'dealer: {" ".join(map(str, dealer_cards))} ({s1})')
                if result == 1:
                    print('end: win')
                    r = 1
                elif result == -1:
                    print('end: lose')
                    r = -1
                else:
                    print('end: draw')
                is_end = True

            trajectory.append(r)
            if is_end: break

        print(f'trajectory: {trajectory}')
        return trajectory

def wireplot(ax, Z):
    w = Z.shape[0]
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, .5, 1]))
    x = y = np.arange(w)
    X, Y = np.meshgrid(x, y)
    ax.plot_wireframe(X, Y, Z[X,Y], lw=1, rstride=1, cstride=1, color='k')
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.set_xlim((0, w-1))
    ax.set_ylim((0, w-1))
    ax.set_zlim((-1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

def step_contour(ax, mat):
    w = mat.shape[0]
    x = range(w)
    y = mat.argmax(axis=-1)
    ax.step(x, y, 'k', linewidth=1, where='mid')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlim((0, w-1))
    ax.set_ylim((-1, w-1))
    ax.set_xticks(range(w))
    ax.set_yticks(range(w+1))
    ax.set_xticklabels(['A']+list(range(2,w+1)))
    ax.set_yticklabels(range(11, 22))
    ax.yaxis.tick_right()
    [s.set_linewidth(1.5) for s in ax.spines.values()]

def fig_5_1():
    total_episodes = 500_000
    game = BlackJack()
    player = FixedStickAgent(20)

    fig = plt.figure(figsize=(10, 8))

    states_usable_ace = np.zeros((10, 10))
    states_usable_ace_count = np.ones((10, 10)) # initialze counts to 1 to avoid 0 being divided
    states_no_usable_ace = np.zeros((10, 10))
    states_no_usable_ace_count = np.ones((10, 10))

    # First-visit MC prediction, i.e. Monte Carlo Sample with On-Policy
    # for episode in range(total_episodes):
    for episode in tqdm(range(total_episodes), leave=False):
        # Generate an episode on policy π: S0, A0, R1, S1, A1, R2, ..., S_{t-1}, A_{t-1}, R_t
        trajectory = game.simulate(player)
        # G <- G + R_{t+1}
        G = trajectory[-1] # for the R_t is always 0
        # loop for each step, note that S_t never appears in S0, S1, ..., S_{t-1}
        for i in range(len(trajectory)//4):
            S, usable_ace, *_ = trajectory[i*4:(i+1)*4]
            S = ((S[0]-1)%10, S[1]-12)
            if usable_ace:
                states_usable_ace[S] += G
                states_usable_ace_count[S] += 1
            else:
                states_no_usable_ace[S] += G
                states_no_usable_ace_count[S] += 1

        V_usable_ace = states_usable_ace / states_usable_ace_count
        V_no_usable_ace = states_no_usable_ace / states_no_usable_ace_count

        if episode == 10_000:
            ax1 = fig.add_subplot(2, 2, 1, projection='3d')
            wireplot(ax1, V_usable_ace)
            ax1.text(-2, 5, 4, 'After 10,000 episodes', fontsize=18)
            ax1.text(-6, 0, -1, 'Usable\nace', fontsize=15, horizontalalignment='center')

            ax2 = fig.add_subplot(2, 2, 3, projection='3d')
            wireplot(ax2, V_no_usable_ace)
            ax2.text(-6, 0, -1, 'No\nusable\nace', fontsize=15, horizontalalignment='center')

    V_usable_ace = states_usable_ace / states_usable_ace_count
    V_no_usable_ace = states_no_usable_ace / states_no_usable_ace_count
    ax1 = fig.add_subplot(2, 2, 2, projection='3d')
    wireplot(ax1, V_usable_ace)
    ax1.text(-2.5, 5, 4, 'After 500,000 episodes', fontsize=18)
            
    ax2 = fig.add_subplot(2, 2, 4, projection='3d')
    wireplot(ax2, V_no_usable_ace)
    ax2.set_xlabel('Dealer showing', fontsize=12)
    ax2.set_ylabel('Player sum', fontsize=12)
    ax2.set_xticks([0, 9])
    ax2.set_yticks([0, 9])
    ax2.set_zticks([-1, 1])
    ax2.set_xticklabels(['A', 10])
    ax2.set_yticklabels([12, 21])

    plt.savefig('fig_5.1_state_value_for_balckjack.png', dpi=300)
    plt.close()

def fig_5_2():
    np.random.seed(2019)
    total_episodes = 500_000
    game = BlackJack()
    player = MCESAgent()

    for episode in tqdm(range(total_episodes), leave=False):
        init = [(np.random.choice(range(2, 12)), np.random.choice(range(12, 22))),
                bool(np.random.choice([0, 1])),
                np.random.choice([ACTION_HIT, ACTION_STICK])]
        player.update(game.simulate(player, init=init))

    Q = player.action_state_value / player.action_state_count
    V = Q.max(axis=-1)

    fig = plt.figure(figsize=(10, 8))
    from matplotlib import gridspec
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 2])
    ax1 = plt.subplot(gs[0])
    step_contour(ax1, player.π[:,:,1])
    ax1.set_title('$\mathcal{\pi}_∗$', fontsize=37, pad=30)
    ax1.text(-2, 4, 'Usable\nace', fontsize=15, horizontalalignment='center')
    ax1.text(3.2, 8, 'STICK', fontsize=15)
    ax1.text(3.5, 3, 'HIT', fontsize=15)

    ax2 = plt.subplot(gs[1], projection='3d')
    wireplot(ax2, V[:,:,1])
    ax2.set_title('$\mathcal{V}_∗$', fontsize=30, pad=50)

    ax3 = plt.subplot(gs[2])
    step_contour(ax3, player.π[:,:,0])
    ax3.text(-2, 3.5, 'No\nUsable\nace', fontsize=15, horizontalalignment='center')
    ax3.text(3.2, 6, 'STICK', fontsize=15)
    ax3.text(7, 2, 'HIT', fontsize=15)

    ax4 = plt.subplot(gs[3], projection='3d')
    wireplot(ax4, V[:,:,0])
    ax4.set_xlabel('Dealer showing', fontsize=12)
    ax4.set_ylabel('Player sum', fontsize=12)
    ax4.set_xticks([0, 9])
    ax4.set_yticks([0, 9])
    ax4.set_zticks([-1, 1])
    ax4.set_xticklabels(['A', 10])
    ax4.set_yticklabels([12, 21])

    plt.savefig('fig_5.2_balckjack_MCES.png', dpi=300, bbox_inches='tight', pad_inches=.5)
    plt.close()

def fig_5_3():
    runs = 100
    episodes = 10_000
    game = BlackJack()
    behavior = RandomAgent()
    target = FixedStickAgent(20)

    init = [(2, 13), True]
    V = -0.27726

    ordinary_imptsamp = np.zeros((runs, episodes))
    weighted_imptsamp = np.zeros((runs, episodes))

    for run in tqdm(range(runs), leave=False):
        ρs = []
        rets = []
        for episode in range(episodes):
            trajectory = game.simulate(behavior, init=init)

            π = 1
            b = 1
            for i in range(len(trajectory)//4):
                S, usable_ace, A, _ = trajectory[i*4:(i+1)*4]
                if A == target.act(S, usable_ace):
                    b *= 0.5
                else:
                    π = 0
            ρ = π / b

            ρs.append(ρ)
            rets.append(trajectory[-1])

        ρs = np.array(ρs)
        rets = np.array(rets)
        wgt_rets = ρs * rets
        accum_wgt_rets = np.add.accumulate(wgt_rets)
        accum_ρs = np.add.accumulate(ρs)

        ordinary_imptsamp[run] = accum_wgt_rets / np.arange(1, episodes+1)
        with np.errstate(divide='ignore',invalid='ignore'):
            weighted_imptsamp[run] = np.where(accum_ρs != 0, accum_wgt_rets / accum_ρs, 0)

    avg_ordimptsamp = ((ordinary_imptsamp - V)**2).mean(0)
    avg_wgtimptsamp = ((weighted_imptsamp - V)**2).mean(0)

    ax = plt.subplot()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(avg_wgtimptsamp, 'r-', lw=1)
    plt.text(1, .55, 'Weighted importance sampling', color='r', fontsize=11)
    ax.plot(avg_ordimptsamp, 'g-')
    plt.text(8, 2.1, 'Ordinary\nimportance\nsampling', color='g', fontsize=11, horizontalalignment='center')
    ax.set_xscale('log')
    ax.set_xlabel('Episodes (log scale)', fontsize=14)
    ax.set_ylabel('Mean\nsquare\nerror\n(average over\n100 runs)', fontsize=14, rotation=0, labelpad=40, verticalalignment='center')
    ax.set_ylim((-0.2, 4))
    ax.set_yticks([0, 2, 4])
    ax.set_xlim((.6, 10000))
    plt.savefig('fig_5.3_ordinary_and_weighted_importance_sampling.png', dpi=300, bbox_inches='tight')
    plt.close()

# fig_5_1()
# fig_5_2()
# fig_5_3()
