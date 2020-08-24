# Chapter 8<br>Planning and Learning with<br>Tabular Methods

### Pseudocode
![](chap8.1-random_sample_q_planning.png)

![](chap8.2-Dyna_Q.png)

![](chap8.4-prioritized_sweeping.png)

![](chap8.5-backup_diagram.png)

### Figure
![](fig_8.2_dyna_maze.png)
**Figure 8.2:** A simple maze (inset) and the average learning curves for Dyna-Q agents varying in their number of planning steps (n) per real step. The task is to travel from S to G as quickly as possible.

![](fig_8.3_dyna_maze_policy.png)
**Figure 8.3:** Policies found by planning and nonplanning Dyna-Q agents halfway through the second episode. The arrows indicate the greedy action in each state; if no arrow is shown for a state, then all of its action values were equal. The black square indicates the location of the agent.

![](fig_8.4_dyna_on_blocking_maze.png)
**Figure 8.4:** Average performance of Dyna agents on a blocking task. The left environment was used for the first 1000 steps, the right environment for the rest. Dyna-Q+ is Dyna-Q with an exploration bonus that encourages exploration.

![](fig_8.5_dyna_on_shortcut_maze.png)
**Figure 8.5:** Average performance of Dyna agents on a shortcut task. The left environment was used for the first 3000 steps, the right environment for the rest.

![](exp_8.4_prioritized_sweeping.png)
**Example 8.4:** Prioritized Sweeping on Mazes

![](fig_8.7_expected_vs_sample_update.png)
**Figure 8.7:** Comparison of efficiency of expected and sample updates.

![](fig_8.8_trajectory_sampling.png)
**Figure 8.8:** Relative efficiency of updates distributed uniformly across the state space versus focused on simulated on-policy trajectories, each starting in the same state. Results are for randomly generated tasks of two sizes and various branching factors, b.