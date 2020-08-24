'''
One state MDP

Author: Urinx
'''
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

LEFT = 0
RIGHT = 1

def simulate():
    π = 1
    b = 1
    while 1:
        a = np.random.choice([LEFT, RIGHT])
        b *= 0.5

        if a == RIGHT:
            π = r = 0
            break
        else:
            if np.random.rand() < 0.1:
                r = 1
                break
    ρ = π / b
    return ρ, r

def fig_5_4():
    # Need around 11 hours and more than 30G memory.
    episodes = 100_000_000
    runs = 10
    ordinary_imptsamp = [] # the ordinary_imptsamp array take around 7.5G space.

    for run in range(runs):
        wgt_rets = []

        for e in tqdm(range(episodes), leave=True):
            ρ, r = simulate()
            wgt_rets.append(ρ * r)

        accum_wgt_rets = np.add.accumulate(wgt_rets)
        ordinary_imptsamp.append(accum_wgt_rets / np.arange(1, episodes+1))

    plt.figure(figsize=(10, 5))
    ax = plt.subplot()
    for i in range(len(ordinary_imptsamp)):
        ax.plot(ordinary_imptsamp[i], lw=1)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xscale('log')
    ax.set_ylim((-0.2, 3))
    ax.set_yticks([0, 1, 2])
    ax.set_xlabel('Episodes (log scale)')
    ax.set_ylabel('Monte-Carlo\nestimate of\n$v_\pi(s)$ with\nordinary\nimportance\nsampling\n(ten runs)', rotation=0)
    ax.yaxis.set_label_coords(-0.1, 0.3)
    plt.savefig('fig_5.4_infinite_variance.png', dpi=300, bbox_inches='tight')
    plt.close()

fig_5_4()
